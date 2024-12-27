#include "lidar_selection.h"

namespace lidar_selection {

//初始化R P W H HTH不知道是什么矩阵
LidarSelector::LidarSelector(const int gridsize, SparseMap* sparsemap ): grid_size(gridsize), sparse_map(sparsemap)
{
    downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
    G = Matrix<double, DIM_STATE, DIM_STATE>::Zero();
    H_T_H = Matrix<double, DIM_STATE, DIM_STATE>::Zero();
    Rli = M3D::Identity();
    Rci = M3D::Identity();
    Rcw = M3D::Identity();
    Jdphi_dR = M3D::Identity();
    Jdp_dt = M3D::Identity();
    Jdp_dR = M3D::Identity();
    Pli = V3D::Zero();
    Pci = V3D::Zero();
    Pcw = V3D::Zero();
    width = 800;
    height = 600;
}

LidarSelector::~LidarSelector() 
{
    delete sparse_map;
    delete sub_sparse_map;
    delete[] grid_num;
    delete[] map_index;
    delete[] map_value;
    delete[] align_flag;
    delete[] patch_cache;
    unordered_map<int, Warp*>().swap(Warp_map);
    unordered_map<VOXEL_KEY, float>().swap(sub_feat_map);
    unordered_map<VOXEL_KEY, VOXEL_POINTS*>().swap(feat_map);  
}

//Lidar to imu外参
void LidarSelector::set_extrinsic(const V3D &transl, const M3D &rot)
{
    Pli = -rot.transpose() * transl;
    Rli = rot.transpose();
}

//初始化
void LidarSelector::init()
{
    sub_sparse_map = new SubSparseMap;
    // 相机 to IMU 外参
    Rci = sparse_map->Rcl * Rli;
    Pci= sparse_map->Rcl*Pli + sparse_map->Pcl;
    M3D Ric;
    V3D Pic;
    Jdphi_dR = Rci;
    Pic = -Rci.transpose() * Pci;
    M3D tmp;
    tmp << SKEW_SYM_MATRX(Pic);
    Jdp_dR = -Rci * tmp;
    width = cam->width();
    height = cam->height();
    grid_n_width = static_cast<int>(width/grid_size);
    grid_n_height = static_cast<int>(height/grid_size);
    length = grid_n_width * grid_n_height;
    fx = cam->errorMultiplier2();
    fy = cam->errorMultiplier() / (4. * fx);
    grid_num = new int[length];
    map_index = new int[length];
    map_value = new float[length];
    display_lidar_map_value = new float[length];
    visual_score = new float[length];
    point_distance = new float[length];
    align_flag = new int[length];
    map_dist = (float*)malloc(sizeof(float)*length);
    memset(grid_num, TYPE_UNKNOWN, sizeof(int)*length);
    memset(map_index, 0, sizeof(int)*length);
    memset(map_value, 0, sizeof(float)*length);
    memset(visual_score, 0, sizeof(float)*length);
    fill_n(point_distance, length, 10000);
    voxel_points_.reserve(length);
    add_voxel_points_.reserve(length);
    visual_points_.reserve(length);
    display_lidar_points_.reserve(length);
    count_img = 0;
    patch_size_total = patch_size * patch_size;
    patch_size_half = static_cast<int>(patch_size/2);
    patch_cache = new float[patch_size_total];
    stage_ = STAGE_FIRST_FRAME;
    pg_down.reset(new PointCloudXYZI());
    Map_points.reset(new PointCloudXYZI());
    Map_points_output.reset(new PointCloudXYZI());
    weight_scale_ = 10;
    weight_function_.reset(new vk::robust_cost::HuberWeightFunction());
    // weight_function_.reset(new vk::robust_cost::TukeyWeightFunction());
    scale_estimator_.reset(new vk::robust_cost::UnitScaleEstimator());
    // scale_estimator_.reset(new vk::robust_cost::MADScaleEstimator());
}

void LidarSelector::reset_grid()
{
    //重置grid_num，用来判断是否找到40 * 40网格最优点的
    memset(grid_num, TYPE_UNKNOWN, sizeof(int)*length);
    //重置map_index,用来记录上面最优点的角点相应值
    memset(map_index, 0, sizeof(int)*length);
    memset(visual_score, 0, sizeof(float)*length);
    fill_n(point_distance, length, 10000);
    fill_n(map_dist, length, 10000);
    std::vector<PointPtr>(length).swap(voxel_points_);
    std::vector<V3D>(length).swap(add_voxel_points_);
    // std::vector<V2D>(length).swap(visual_points_);
    voxel_points_.reserve(length);
    add_voxel_points_.reserve(length);
    // visual_points_.reserve(length);
}
//d(u, v) / d(x, y z),J存放偏导数雅可比矩阵
void LidarSelector::dpi(V3D p, MD(2,3)& J) {
    const double x = p[0];
    const double y = p[1];
    const double z_inv = 1./p[2];
    const double z_inv_2 = z_inv * z_inv;
    J(0,0) = fx * z_inv;
    J(0,1) = 0.0;
    J(0,2) = -fx * x * z_inv_2;
    J(1,0) = 0.0;
    J(1,1) = fy * z_inv;
    J(1,2) = -fy * y * z_inv_2;
}
// [fx/z,  0,      - fx * x / z * z
//  0,     fy/z,   fy * y / z * z     
// ]

float LidarSelector::CheckGoodPoints(cv::Mat img, V2D uv)
{
    const float u_ref = uv[0];
    const float v_ref = uv[1];
    const int u_ref_i = floorf(uv[0]); 
    const int v_ref_i = floorf(uv[1]);
    const float subpix_u_ref = u_ref-u_ref_i;
    const float subpix_v_ref = v_ref-v_ref_i;
    uint8_t* img_ptr = (uint8_t*) img.data + (v_ref_i)*width + (u_ref_i);
    float gu = 2*(img_ptr[1] - img_ptr[-1]) + img_ptr[1-width] - img_ptr[-1-width] + img_ptr[1+width] - img_ptr[-1+width];
    float gv = 2*(img_ptr[width] - img_ptr[-width]) + img_ptr[width+1] - img_ptr[-width+1] + img_ptr[width-1] - img_ptr[-width-1];
    return fabs(gu)+fabs(gv);
}

void LidarSelector::getpatch(cv::Mat img, V2D pc, float* patch_tmp, int level) 
{
    const float u_ref = pc[0];
    const float v_ref = pc[1];
    const int scale =  (1<<level);
    const int u_ref_i = floorf(pc[0]/scale)*scale; 
    const int v_ref_i = floorf(pc[1]/scale)*scale;
    const float subpix_u_ref = (u_ref-u_ref_i)/scale;
    const float subpix_v_ref = (v_ref-v_ref_i)/scale;
    const float w_ref_tl = (1.0-subpix_u_ref) * (1.0-subpix_v_ref);
    const float w_ref_tr = subpix_u_ref * (1.0-subpix_v_ref);
    const float w_ref_bl = (1.0-subpix_u_ref) * subpix_v_ref;
    const float w_ref_br = subpix_u_ref * subpix_v_ref;
    for (int x=0; x<patch_size; x++) 
    {
        uint8_t* img_ptr = (uint8_t*) img.data + (v_ref_i-patch_size_half*scale+x*scale)*width + (u_ref_i-patch_size_half*scale);
        for (int y=0; y<patch_size; y++, img_ptr+=scale)
        {
            patch_tmp[patch_size_total*level+x*patch_size+y] = w_ref_tl*img_ptr[0] + w_ref_tr*img_ptr[scale] + w_ref_bl*img_ptr[scale*width] + w_ref_br*img_ptr[scale*width+scale];
        }
    }
}

//把雷达点加入视觉地图？
//把一个grid内最高梯度的点加入视觉地图
void LidarSelector::addSparseMap(cv::Mat img, PointCloudXYZI::Ptr pg) 
{
    //其实相当于在这里改代码
    // double t0 = omp_get_wtime();
    reset_grid();

    std::vector<V2D>(length).swap(visual_points_);
    visual_points_.reserve(length);
    memset(display_lidar_map_value, 0, sizeof(float)*length);
    std::vector<V2D>(length).swap(display_lidar_points_);
    display_lidar_points_.reserve(length);
    

    vector<cv::Point2f> n_pts;

    //后面的block size可以修改。
    cv::goodFeaturesToTrack(img, n_pts, 64, 0.01, 30);
	//指定亚像素计算迭代标注
	cv::TermCriteria criteria = cv::TermCriteria(
					cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS,
					40,
					0.01);
 
	//亚像素检测
	cv::cornerSubPix(img, n_pts, cv::Size(5, 5), cv::Size(-1, -1), criteria);
    // ROS_ERROR("n_pts.size() = %d\n", n_pts.size());
	for (int i = 0; i < n_pts.size(); i++)
	{
        V2D temp(n_pts[i].x, n_pts[i].y);
        if(new_frame_->cam_->isInFrame(temp.cast<int>(), (patch_size_half+1)*8))
        {
            int index = static_cast<int>(n_pts[i].x/grid_size)*grid_n_height + static_cast<int>(n_pts[i].y/grid_size);
            float cur_value = vk::shiTomasiScore(img, n_pts[i].x, n_pts[i].y);
            if(cur_value > visual_score[index]) {
                visual_score[index] = cur_value;
                visual_points_[index] = temp;
            }
        }
		cv::circle(img, n_pts[i], 5, cv::Scalar(0, 255, 0), 2, 8, 0);
	}
 
	// cv::imshow("corner", img);

    // cv::waitKey(50);

    // double t_b1 = omp_get_wtime() - t0;
    // t0 = omp_get_wtime();
    std::cout << "1111111 addSparseMap pg.size = " << pg->points.size() << std::endl;
    //pg是没有下采样的上一帧世界点
    for (int i=0; i<pg->size(); i++) 
    {
        V3D pt(pg->points[i].x, pg->points[i].y, pg->points[i].z);
        //相机坐标系
        //这里pc只是一个二维变量，跟像素没关系。
        // 把雷达世界点转到相机坐标下的点，这里用的SE3是通过前面的IMU update的。
        V2D pc(new_frame_->w2c(pt));
        //40 * 40的网格，检查是否在相机视野内，考虑边缘内40为边界，因为雷达投过去不一定在图像中
        if(new_frame_->cam_->isInFrame(pc.cast<int>(), (patch_size_half+1)*8)) // 20px is the patch size in the matcher
        {
            //grid size 40 
            int index = static_cast<int>(pc[0]/grid_size)*grid_n_height + static_cast<int>(pc[1]/grid_size);
        
            // float cur_value = CheckGoodPoints(img, pc);
            //计算shiTomasiScore角点响应值
            //这里应该是有像素信息的，根据像素信息来计算角点响应值
            //这个就是goodFeaturesToTrack用的方法。
            float cur_value = vk::shiTomasiScore(img, pc[0], pc[1]);
            if(cur_value > display_lidar_map_value[index]) {
                display_lidar_map_value[index] = cur_value;
                display_lidar_points_[index] = pc;
            }

            // if (cur_value > map_value[index]) //&& (grid_num[index] != TYPE_MAP || map_value[index]<=10)) //! only add in not occupied grid
            // {
            //     //对应文章中40 * 40网格找最小的深度点，这里用的是角点响应值
            //     map_value[index] = cur_value;
            //     add_voxel_points_[index] = pt;
            //     grid_num[index] = TYPE_POINTCLOUD;
            // }
            if(visual_score[index] == 0) {
                // ROS_ERROR("visual_score = 0, and index = %d", index);
                continue;
            }
            auto vec = visual_points_[index] - pc;
            if(vec.norm() < point_distance[index]) {
                point_distance[index] = vec.norm();
                map_value[index] = cur_value;
                add_voxel_points_[index] = pt;
                grid_num[index] = TYPE_POINTCLOUD;
            }
        }
    }

    // double t_b2 = omp_get_wtime() - t0;
    // t0 = omp_get_wtime();
    
    int add=0;
    //遍历所有网格
    for (int i=0; i<length; i++) 
    {
        //这里的i就是与上面的index相关，即每个40*40网格找到了角度响应值最大的点
        if (grid_num[i]==TYPE_POINTCLOUD)// && (map_value[i]>=10)) //! debug
        {
            //要添加到视觉子图的点，将3D转换到像素平面点
            V3D pt = add_voxel_points_[i];
            V2D pc(new_frame_->w2c(pt));
            float* patch = new float[patch_size_total*3];
            //获取中心点的一块8 *8 的patch，应该是放在了patch里面
            getpatch(img, pc, patch, 0);
            getpatch(img, pc, patch, 1);
            getpatch(img, pc, patch, 2);
            //pt_new需要添加的点，3D
            PointPtr pt_new(new Point(pt));
            //pc换到单位球中的坐标
            Vector3d f = cam->cam2world(pc);
            //创建一个新的feature，level是0层，把pose赋值了,patch也关联了
            //一个视觉世界点point，关联了当前图像的一块patch，以pc为中心，同时有这个
            FeaturePtr ftr_new(new Feature(patch, pc, f, new_frame_->T_f_w_, map_value[i], 0));
            //把feature的img设置
            ftr_new->img = new_frame_->img_pyr_[0];
            // ftr_new->ImgPyr.resize(5);
            // for(int i=0;i<5;i++) ftr_new->ImgPyr[i] = new_frame_->img_pyr_[i];
            ftr_new->id_ = new_frame_->id_;

            pt_new->addFrameRef(ftr_new);
            //角度响应值
            pt_new->value = map_value[i];
            AddPoint(pt_new);
            //看log好像每次添加的点并不多
            add += 1;
        }
    }

    // double t_b3 = omp_get_wtime() - t0;
    if(add < 20) {
        ROS_ERROR("[ VIO ]: Add %d 3D points.888888888888888888888888888888888888888\n", add);
    }
    // double t_b1 = omp_get_wtime() - t0;
    printf("[ VIO ]: Add %d 3D points.\n", add);
    // printf("pg.size: %d \n", pg->size());
    // printf("B1. : %.6lf \n", t_b1);
    // printf("B2. : %.6lf \n", t_b2);
    // printf("B3. : %.6lf \n", t_b3);
}

//feat_map是全局地图
//传进来的已经是40 * 40网格最优的点，还要用体素减少检索时间
void LidarSelector::AddPoint(PointPtr pt_new)
{
    //世界坐标系中的点坐标
    V3D pt_w(pt_new->pos_[0], pt_new->pos_[1], pt_new->pos_[2]);
    double voxel_size = 0.5;
    float loc_xyz[3];
    for(int j=0; j<3; j++)
    {
        //3D点是单位m，那么这里voxel_size是0.5m,所有，原有点会是0.5m的多少倍，比如，小于0.5m的都是0
      loc_xyz[j] = pt_w[j] / voxel_size;
      if(loc_xyz[j] < 0) //小于0，是指相对原点后撤了吗
      {
        loc_xyz[j] -= 1.0;
      }
    }
    //key
    VOXEL_KEY position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    auto iter = feat_map.find(position);
    //feat_map是全局地图，而且这里的点是自定义的视觉地图点。
    //如果体素地图中有点，直接在map中的voxel_points添加pt就行。
    if(iter != feat_map.end())
    {
      iter->second->voxel_points.push_back(pt_new);
      iter->second->count++;
    }
    else //如果之前每在这个体素插过点，新建一个体素，在里面添加点，然后加入map中
    {
      VOXEL_POINTS *ot = new VOXEL_POINTS(0);
      ot->voxel_points.push_back(pt_new);
      feat_map[position] = ot;
    }
}

void LidarSelector::getWarpMatrixAffine(
    const vk::AbstractCamera& cam,
    const Vector2d& px_ref,
    const Vector3d& f_ref,
    const double depth_ref,
    const SE3& T_cur_ref,
    const int level_ref,    // the corresponding pyrimid level of px_ref
    const int pyramid_level,
    const int halfpatch_size,
    Matrix2d& A_cur_ref)
{
  // Compute affine warp matrix A_ref_cur
  const Vector3d xyz_ref(f_ref*depth_ref);
  Vector3d xyz_du_ref(cam.cam2world(px_ref + Vector2d(halfpatch_size,0)*(1<<level_ref)*(1<<pyramid_level)));
  Vector3d xyz_dv_ref(cam.cam2world(px_ref + Vector2d(0,halfpatch_size)*(1<<level_ref)*(1<<pyramid_level)));
//   Vector3d xyz_du_ref(cam.cam2world(px_ref + Vector2d(halfpatch_size,0)*(1<<level_ref)));
//   Vector3d xyz_dv_ref(cam.cam2world(px_ref + Vector2d(0,halfpatch_size)*(1<<level_ref)));
  xyz_du_ref *= xyz_ref[2]/xyz_du_ref[2];
  xyz_dv_ref *= xyz_ref[2]/xyz_dv_ref[2];
  const Vector2d px_cur(cam.world2cam(T_cur_ref*(xyz_ref)));
  const Vector2d px_du(cam.world2cam(T_cur_ref*(xyz_du_ref)));
  const Vector2d px_dv(cam.world2cam(T_cur_ref*(xyz_dv_ref)));
  A_cur_ref.col(0) = (px_du - px_cur)/halfpatch_size;
  A_cur_ref.col(1) = (px_dv - px_cur)/halfpatch_size;
}

void LidarSelector::warpAffine(
    const Matrix2d& A_cur_ref,
    const cv::Mat& img_ref,
    const Vector2d& px_ref,
    const int level_ref,
    const int search_level,
    const int pyramid_level,
    const int halfpatch_size,
    float* patch)
{
  const int patch_size = halfpatch_size*2 ;
  const Matrix2f A_ref_cur = A_cur_ref.inverse().cast<float>();
  if(isnan(A_ref_cur(0,0)))
  {
    printf("Affine warp is NaN, probably camera has no translation\n"); // TODO
    return;
  }
//   Perform the warp on a larger patch.
//   float* patch_ptr = patch;
//   const Vector2f px_ref_pyr = px_ref.cast<float>() / (1<<level_ref) / (1<<pyramid_level);
//   const Vector2f px_ref_pyr = px_ref.cast<float>() / (1<<level_ref);
  for (int y=0; y<patch_size; ++y)
  {
    for (int x=0; x<patch_size; ++x)//, ++patch_ptr)
    {
      // P[patch_size_total*level + x*patch_size+y]
      Vector2f px_patch(x-halfpatch_size, y-halfpatch_size);
      px_patch *= (1<<search_level);
      px_patch *= (1<<pyramid_level);
      const Vector2f px(A_ref_cur*px_patch + px_ref.cast<float>());
      if (px[0]<0 || px[1]<0 || px[0]>=img_ref.cols-1 || px[1]>=img_ref.rows-1)
        patch[patch_size_total*pyramid_level + y*patch_size+x] = 0;
        // *patch_ptr = 0;
      else
        patch[patch_size_total*pyramid_level + y*patch_size+x] = (float) vk::interpolateMat_8u(img_ref, px[0], px[1]);
        // *patch_ptr = (uint8_t) vk::interpolateMat_8u(img_ref, px[0], px[1]);
    }
  }
}

double LidarSelector::NCC(float* ref_patch, float* cur_patch, int patch_size)
{    
    double sum_ref = std::accumulate(ref_patch, ref_patch + patch_size, 0.0);
    double mean_ref =  sum_ref / patch_size;

    double sum_cur = std::accumulate(cur_patch, cur_patch + patch_size, 0.0);
    double mean_curr =  sum_cur / patch_size;

    double numerator = 0, demoniator1 = 0, demoniator2 = 0;
    for (int i = 0; i < patch_size; i++) 
    {
        double n = (ref_patch[i] - mean_ref) * (cur_patch[i] - mean_curr);
        numerator += n;
        demoniator1 += (ref_patch[i] - mean_ref) * (ref_patch[i] - mean_ref);
        demoniator2 += (cur_patch[i] - mean_curr) * (cur_patch[i] - mean_curr);
    }
    return numerator / sqrt(demoniator1 * demoniator2 + 1e-10);
}

int LidarSelector::getBestSearchLevel(
    const Matrix2d& A_cur_ref,
    const int max_level)
{
  // Compute patch level in other image
  int search_level = 0;
  double D = A_cur_ref.determinant();

  while(D > 3.0 && search_level < max_level)
  {
    search_level += 1;
    D *= 0.25;
  }
  return search_level;
}

void LidarSelector::createPatchFromPatchWithBorder(float* patch_with_border, float* patch_ref)
{
  float* ref_patch_ptr = patch_ref;
  for(int y=1; y<patch_size+1; ++y, ref_patch_ptr += patch_size)
  {
    float* ref_patch_border_ptr = patch_with_border + y*(patch_size+2) + 1;
    for(int x=0; x<patch_size; ++x)
      ref_patch_ptr[x] = ref_patch_border_ptr[x];
  }
}
//pg难道是最近的雷达点，或者简称为上一帧雷达点。
//这里找地图点，应该不用改
void LidarSelector::addFromSparseMap(cv::Mat img, PointCloudXYZI::Ptr pg)
{
    //unordered_map<VOXEL_KEY, VOXEL_POINTS*> feat_map
    if(feat_map.size()<=0) return;
    // double ts0 = omp_get_wtime();
    // PointCloudXYZI点云 pg_down
    //feat_map是全局视觉地图，size是map的size，并不是真正点的数量
    pg_down->reserve(feat_map.size());
    //这里应该是降采样，根据体素来
    downSizeFilter.setInputCloud(pg);
    downSizeFilter.filter(*pg_down);

    std::cout << "!!!!!!!feat_map size = " << feat_map.size() << 
    ",  pg_down size = " << pg_down->points.size() << "!!!!!!!" << std::endl;
    
    reset_grid();
    memset(map_value, 0, sizeof(float)*length);

    //清空
    sub_sparse_map->reset();
    deque< PointPtr >().swap(sub_map_cur_frame_); //deque< PointPtr >

    float voxel_size = 0.5;
    
    unordered_map<VOXEL_KEY, float>().swap(sub_feat_map); //unordered_map<VOXEL_KEY, float>
    unordered_map<int, Warp*>().swap(Warp_map);

    // cv::Mat depth_img = cv::Mat::zeros(height, width, CV_32FC1);
    // float* it = (float*)depth_img.data;

    float it[height*width] = {0.0};

    double t_insert, t_depth, t_position;
    t_insert=t_depth=t_position=0;

    int loc_xyz[3];

    // printf("A0. initial depthmap: %.6lf \n", omp_get_wtime() - ts0);
    // double ts1 = omp_get_wtime();

    //pg_down是不是就是点云地图呢

    //pg_down降采样后的
    for(int i=0; i<pg_down->size(); i++)
    {
        // Transform Point to world coordinate
        V3D pt_w(pg_down->points[i].x, pg_down->points[i].y, pg_down->points[i].z);

        // Determine the key of hash table      
        for(int j=0; j<3; j++)
        {
            //voxel_size是0.5
            //xyz单位是m, voxel_size是0.5，所以loc_xyz是0.5的倍数，最后归一到以0.5为刻度的坐标上。
            loc_xyz[j] = floor(pt_w[j] / voxel_size);
        }
        //得到一个voxel key
        VOXEL_KEY position(loc_xyz[0], loc_xyz[1], loc_xyz[2]);

        //相当于从地图里找点的操作
        //这个sub_feat_map的含义：
        //value是float
        auto iter = sub_feat_map.find(position);
        //这里应该就是插入的操作了，把每个体素的position置为1
        if(iter == sub_feat_map.end())
        {
            sub_feat_map[position] = 1.0;
        }
                    
        //把世界坐标系的点换成相机坐标系的点
        //还是三维点
        V3D pt_c(new_frame_->w2f(pt_w));

        V2D px;
        // 投影后在前面？
        if(pt_c[2] > 0)
        {
            //变成像素坐标点
            px[0] = fx * pt_c[0]/pt_c[2] + cx;
            px[1] = fy * pt_c[1]/pt_c[2] + cy;

            //检查图像是否越界，考虑是边缘40以内为边界
            if(new_frame_->cam_->isInFrame(px.cast<int>(), (patch_size_half+1)*8))
            {
                float depth = pt_c[2];
                int col = int(px[0]);
                int row = int(px[1]);
                it[width*row+col] = depth;        
            }
        }
    }
    
    // imshow("depth_img", depth_img);
    // printf("A1: %.6lf \n", omp_get_wtime() - ts1);
    // printf("A11. calculate pt position: %.6lf \n", t_position);
    // printf("A12. sub_postion.insert(position): %.6lf \n", t_insert);
    // printf("A13. generate depth map: %.6lf \n", t_depth);
    // printf("A. projection: %.6lf \n", omp_get_wtime() - ts0);
    

    // double t1 = omp_get_wtime();

    std::cout << "111111pg_down_size = " << pg_down->size() << 
    ", sub_feat_map size = " << sub_feat_map.size() << "@1111111" << std::endl;

    //feat_Map应该是视觉全局地图
    for(auto& iter : sub_feat_map)
    {   
        VOXEL_KEY position = iter.first;
        // double t4 = omp_get_wtime();
        //从全局地图里找到
        auto corre_voxel = feat_map.find(position);
        // double t5 = omp_get_wtime();

        if(corre_voxel != feat_map.end())
        {
            std::vector<PointPtr> &voxel_points = corre_voxel->second->voxel_points;
            int voxel_num = voxel_points.size();
            //在这个体素里有多少个点
            for (int i=0; i<voxel_num; i++)
            {
                PointPtr pt = voxel_points[i];
                if(pt==nullptr) continue;
                //把地图点的坐标，换到当前相机坐标系
                V3D pt_cam(new_frame_->w2f(pt->pos_));
                if(pt_cam[2]<0) continue;

                //二维点
                V2D pc(new_frame_->w2c(pt->pos_));

                FeaturePtr ref_ftr;
                //40 * 40的网格，检查是否在相机视野内，考虑边缘内40为边界，因为雷达投过去不一定在图像中
                if(new_frame_->cam_->isInFrame(pc.cast<int>(), (patch_size_half+1)*8)) // 20px is the patch size in the matcher
                {
                    // 看在那个grid里
                    int index = static_cast<int>(pc[0]/grid_size)*grid_n_height + static_cast<int>(pc[1]/grid_size);
                    grid_num[index] = TYPE_MAP;
                    //地图点 到 此时相机坐标原点的距离（世界坐标系下）
                    Vector3d obs_vec(new_frame_->pos() - pt->pos_);

                    float cur_dist = obs_vec.norm(); //三轴平方之和
                    float cur_value = pt->value;

                    //在地图点中找到一个网格最近的，留下
                    //找到最小距离的点，40 *40 网格内 index是网格index
                    if (cur_dist <= map_dist[index]) 
                    {
                        map_dist[index] = cur_dist;
                        voxel_points_[index] = pt;
                    } 

                    //一个网格里面，像素最大的地图点
                    if (cur_value >= map_value[index])
                    {
                        map_value[index] = cur_value;
                    }
                }
            }    
        } 
    }
        
    // double t2 = omp_get_wtime();

    // cout<<"B. feat_map.find: "<<t2-t1<<endl;

    double t_2, t_3, t_4, t_5;
    t_2=t_3=t_4=t_5=0;

    //遍历所有网格，把前面投影的点
    for (int i=0; i<length; i++) 
    { 
        //如果投影成功了
        if (grid_num[i]==TYPE_MAP) //&& map_value[i]>10)
        {
            // double t_1 = omp_get_wtime();


            //dist最小的点
            PointPtr pt = voxel_points_[i];

            if(pt==nullptr) continue;

            //像素坐标点
            V2D pc(new_frame_->w2c(pt->pos_));
            //相机坐标点
            V3D pt_cam(new_frame_->w2f(pt->pos_));
   
            bool depth_continous = false;
            // -4 ~ 4
            for (int u=-patch_size_half; u<=patch_size_half; u++)
            {
                // -4 ~ 4
                for (int v=-patch_size_half; v<=patch_size_half; v++)
                {
                    if(u==0 && v==0) continue;

                    //当前点深度，一个范围 8 * 8
                    float depth = it[width*(v+int(pc[1]))+u+int(pc[0])];

                    if(depth == 0.) continue;
                    // 与地图点深度 差
                    double delta_dist = abs(pt_cam[2]-depth);

                    //深度不连续
                    if(delta_dist > 1.5)
                    {                
                        depth_continous = true;
                        break;
                    }
                }
                if(depth_continous) break;
            }
            //深度不连续点，跳过
            if(depth_continous) continue;

            // t_2 += omp_get_wtime() - t_1;

            // t_1 = omp_get_wtime();
            
            FeaturePtr ref_ftr;

            if(!pt->getCloseViewObs(new_frame_->pos(), ref_ftr, pc)) continue;

            // t_3 += omp_get_wtime() - t_1;

            float* patch_wrap = new float[patch_size_total*3];

            patch_wrap = ref_ftr->patch;

            // t_1 = omp_get_wtime();
           
            int search_level;
            Matrix2d A_cur_ref_zero;

            auto iter_warp = Warp_map.find(ref_ftr->id_);
            //第一次肯定没有
            if(iter_warp != Warp_map.end())
            {
                search_level = iter_warp->second->search_level;
                A_cur_ref_zero = iter_warp->second->A_cur_ref;
            }
            else
            {
                getWarpMatrixAffine(*cam, ref_ftr->px, ref_ftr->f, (ref_ftr->pos() - pt->pos_).norm(), 
                new_frame_->T_f_w_ * ref_ftr->T_f_w_.inverse(), 0, 0, patch_size_half, A_cur_ref_zero);
                
                search_level = getBestSearchLevel(A_cur_ref_zero, 2);

                Warp *ot = new Warp(search_level, A_cur_ref_zero);
                Warp_map[ref_ftr->id_] = ot;
            }

            // t_4 += omp_get_wtime() - t_1;

            // t_1 = omp_get_wtime();

            for(int pyramid_level=0; pyramid_level<=0; pyramid_level++)
            {                
                warpAffine(A_cur_ref_zero, ref_ftr->img, ref_ftr->px, ref_ftr->level, search_level, pyramid_level, patch_size_half, patch_wrap);
            }

            getpatch(img, pc, patch_cache, 0);

            if(ncc_en)
            {
                double ncc = NCC(patch_wrap, patch_cache, patch_size_total);
                if(ncc < ncc_thre) continue;
            }

            float error = 0.0;
            for (int ind=0; ind<patch_size_total; ind++) 
            {
                error += (patch_wrap[ind]-patch_cache[ind]) * (patch_wrap[ind]-patch_cache[ind]);
            }
            if(error > outlier_threshold*patch_size_total) continue;
            
            //这里相当于当前帧的视觉子图
            sub_map_cur_frame_.push_back(pt);

            sub_sparse_map->align_errors.push_back(error);
            sub_sparse_map->propa_errors.push_back(error);
            sub_sparse_map->search_levels.push_back(search_level);
            sub_sparse_map->errors.push_back(error);
            sub_sparse_map->index.push_back(i);  //index
            sub_sparse_map->voxel_points.push_back(pt);
            sub_sparse_map->patch.push_back(patch_wrap);
            // sub_sparse_map->px_cur.push_back(pc);
            // sub_sparse_map->propa_px_cur.push_back(pc);
            // t_5 += omp_get_wtime() - t_1;
        }
    }
    // double t3 = omp_get_wtime();
    // cout<<"C. addSubSparseMap: "<<t3-t2<<endl;
    // cout<<"depthcontinuous: C1 "<<t_2<<" C2 "<<t_3<<" C3 "<<t_4<<" C4 "<<t_5<<endl;
    printf("[ VIO ]: choose %d points from sub_sparse_map.\n", int(sub_sparse_map->index.size()));
}

bool LidarSelector::align2D(
    const cv::Mat& cur_img,
    float* ref_patch_with_border,
    float* ref_patch,
    const int n_iter,
    Vector2d& cur_px_estimate,
    int index)
{
#ifdef __ARM_NEON__
  if(!no_simd)
    return align2D_NEON(cur_img, ref_patch_with_border, ref_patch, n_iter, cur_px_estimate);
#endif

  const int halfpatch_size_ = 4;
  const int patch_size_ = 8;
  const int patch_area_ = 64;
  bool converged=false;

  // compute derivative of template and prepare inverse compositional
  float __attribute__((__aligned__(16))) ref_patch_dx[patch_area_];
  float __attribute__((__aligned__(16))) ref_patch_dy[patch_area_];
  Matrix3f H; H.setZero();

  // compute gradient and hessian
  const int ref_step = patch_size_+2;
  float* it_dx = ref_patch_dx;
  float* it_dy = ref_patch_dy;
  for(int y=0; y<patch_size_; ++y) 
  {
    float* it = ref_patch_with_border + (y+1)*ref_step + 1; 
    for(int x=0; x<patch_size_; ++x, ++it, ++it_dx, ++it_dy)
    {
      Vector3f J;
      J[0] = 0.5 * (it[1] - it[-1]); 
      J[1] = 0.5 * (it[ref_step] - it[-ref_step]); 
      J[2] = 1; 
      *it_dx = J[0];
      *it_dy = J[1];
      H += J*J.transpose(); 
    }
  }
  Matrix3f Hinv = H.inverse();
  float mean_diff = 0;

  // Compute pixel location in new image:
  float u = cur_px_estimate.x();
  float v = cur_px_estimate.y();

  // termination condition
  const float min_update_squared = 0.03*0.03;//0.03*0.03
  const int cur_step = cur_img.step.p[0];
  float chi2 = 0;
  chi2 = sub_sparse_map->propa_errors[index];
  Vector3f update; update.setZero();
  for(int iter = 0; iter<n_iter; ++iter)
  {
    int u_r = floor(u);
    int v_r = floor(v);
    if(u_r < halfpatch_size_ || v_r < halfpatch_size_ || u_r >= cur_img.cols-halfpatch_size_ || v_r >= cur_img.rows-halfpatch_size_)
      break;

    if(isnan(u) || isnan(v)) // TODO very rarely this can happen, maybe H is singular? should not be at corner.. check
      return false;

    // compute interpolation weights
    float subpix_x = u-u_r;
    float subpix_y = v-v_r;
    float wTL = (1.0-subpix_x)*(1.0-subpix_y);
    float wTR = subpix_x * (1.0-subpix_y);
    float wBL = (1.0-subpix_x)*subpix_y;
    float wBR = subpix_x * subpix_y;

    // loop through search_patch, interpolate
    float* it_ref = ref_patch;
    float* it_ref_dx = ref_patch_dx;
    float* it_ref_dy = ref_patch_dy;
    float new_chi2 = 0.0;
    Vector3f Jres; Jres.setZero();
    for(int y=0; y<patch_size_; ++y)
    {
      uint8_t* it = (uint8_t*) cur_img.data + (v_r+y-halfpatch_size_)*cur_step + u_r-halfpatch_size_; 
      for(int x=0; x<patch_size_; ++x, ++it, ++it_ref, ++it_ref_dx, ++it_ref_dy)
      {
        float search_pixel = wTL*it[0] + wTR*it[1] + wBL*it[cur_step] + wBR*it[cur_step+1];
        float res = search_pixel - *it_ref + mean_diff;
        Jres[0] -= res*(*it_ref_dx);
        Jres[1] -= res*(*it_ref_dy);
        Jres[2] -= res;
        new_chi2 += res*res;
      }
    }

    if(iter > 0 && new_chi2 > chi2)
    {
    //   cout << "error increased." << endl;
      u -= update[0];
      v -= update[1];
      break;
    }
    chi2 = new_chi2;

    sub_sparse_map->align_errors[index] = new_chi2;

    update = Hinv * Jres;
    u += update[0];
    v += update[1];
    mean_diff += update[2];

#if SUBPIX_VERBOSE
    cout << "Iter " << iter << ":"
         << "\t u=" << u << ", v=" << v
         << "\t update = " << update[0] << ", " << update[1]
//         << "\t new chi2 = " << new_chi2 << endl;
#endif

    if(update[0]*update[0]+update[1]*update[1] < min_update_squared)
    {
#if SUBPIX_VERBOSE
      cout << "converged." << endl;
#endif
      converged=true;
      break;
    }
  }

  cur_px_estimate << u, v;
  return converged;
}

void LidarSelector::FeatureAlignment(cv::Mat img)
{
    int total_points = sub_sparse_map->index.size();
    if (total_points==0) return;
    memset(align_flag, 0, length);
    int FeatureAlignmentNum = 0;
       
    for (int i=0; i<total_points; i++) 
    {
        bool res;
        int search_level = sub_sparse_map->search_levels[i];
        Vector2d px_scaled(sub_sparse_map->px_cur[i]/(1<<search_level));
        res = align2D(new_frame_->img_pyr_[search_level], sub_sparse_map->patch_with_border[i], sub_sparse_map->patch[i],
                        20, px_scaled, i);
        sub_sparse_map->px_cur[i] = px_scaled * (1<<search_level);
        if(res)
        {
            align_flag[i] = 1;
            FeatureAlignmentNum++;
        }
    }
}

//这里应该也要改，把用于计算J的色块选取方法变了。
//total_residual开始是1e10
float LidarSelector::UpdateState(cv::Mat img, float total_residual, int level) 
{
    int total_points = sub_sparse_map->index.size();
    if (total_points==0) return 0.;
    //旧状态
    StatesGroup old_state = (*state);
    V2D pc; 
    //1 * 2矩阵
    MD(1,2) Jimg;
    //2 * 3矩阵
    MD(2,3) Jdpi;
    //1 * 3矩阵
    MD(1,3) Jdphi, Jdp, JdR, Jdt;
    VectorXd z;
    // VectorXd R;
    bool EKF_end = false;
    /* Compute J */
    float error=0.0, last_error=total_residual, patch_error=0.0, last_patch_error=0.0, propa_error=0.0;
    // MatrixXd H;
    bool z_init = true;
    //点 * patch大小
    const int H_DIM = total_points * patch_size_total;
    std::cout << "-------------total_points: " << total_points << ", H_DIM: " << 
            H_DIM << "-------------" << std::endl;
    MatrixXd H_sub;
    
    // K.resize(H_DIM, H_DIM);
    z.resize(H_DIM);
    z.setZero();
    // R.resize(H_DIM);
    // R.setZero();

    // H.resize(H_DIM, DIM_STATE);
    // H.setZero();
    //MatrixXd
    H_sub.resize(H_DIM, 6);
    H_sub.setZero();

    //迭代状态卡尔曼滤波，有个最大迭代次数，或者当err小于某个值的时候，退出
    for (int iteration=0; iteration<NUM_MAX_ITERATIONS; iteration++) 
    {
        // double t1 = omp_get_wtime();
        double count_outlier = 0;
     
        error = 0.0;
        propa_error = 0.0;
        n_meas_ =0;
        //world to imu(body)
        M3D Rwi(state->rot_end);
        //P world to imu(body)，相当于IMU坐标原点在world坐标系下的位置
        //或者说，是飞机在世界坐标系的位置表示。
        V3D Pwi(state->pos_end);
        //坐标系变换，camera to world 坐标系变换 = camera to imu * imu to world
        Rcw = Rci * Rwi.transpose();
        //世界坐标系原点在相机坐标系中的表示
        Pcw = -Rci*Rwi.transpose()*Pwi + Pci;
        Jdp_dt = Rci * Rwi.transpose();
        
        //p的反对称矩阵
        M3D p_hat;
        int i;
        //对于视觉子图里面的每一个点，与图像计算光度误差
        for (i=0; i<sub_sparse_map->index.size(); i++) 
        {
            patch_error = 0.0;
            int search_level = sub_sparse_map->search_levels[i];
            // 0 1 2
            int pyramid_level = level + search_level;
            // std::cout << "8888888pyramid_level: " << pyramid_level << "888888" << std::endl;
            //不同level scale不一样，level3就是 8
            // 1 2 4
            const int scale =  (1<<pyramid_level);
            // std::cout << "8888888_scale: " << scale << "888888" << std::endl;
            PointPtr pt = sub_sparse_map->voxel_points[i];

            if(pt==nullptr) continue;

            //算相机坐标系下的点
            //点的投影是左乘一个camera to world，转到当前相机坐标系
            V3D pf = Rcw * pt->pos_ + Pcw;
            //相机坐标系到像素坐标
            pc = cam->world2cam(pf);
            // if((level==2 && iteration==0) || (level==1 && iteration==0) || level==0)
            {
                dpi(pf, Jdpi);
                //把变量转成反对称矩阵
                p_hat << SKEW_SYM_MATRX(pf);
            }

            //双线性插值计算
            const float u_ref = pc[0];
            const float v_ref = pc[1];
            const int u_ref_i = floorf(pc[0]/scale)*scale; 
            const int v_ref_i = floorf(pc[1]/scale)*scale;
            //获取亚像素部分
            const float subpix_u_ref = (u_ref-u_ref_i)/scale;
            const float subpix_v_ref = (v_ref-v_ref_i)/scale;
            const float w_ref_tl = (1.0-subpix_u_ref) * (1.0-subpix_v_ref);
            const float w_ref_tr = subpix_u_ref * (1.0-subpix_v_ref);
            const float w_ref_bl = (1.0-subpix_u_ref) * subpix_v_ref;
            const float w_ref_br = subpix_u_ref * subpix_v_ref;
            
            //图像金字塔层次上的特征匹配和优化
            //这段代码通过双线性插值计算图像金字塔层次上的特征点梯度，
            // 然后使用这些梯度计算与旋转和位移相关的雅可比矩阵，
            // 最后计算特征点的残差并更新Hessian矩阵。这是图像处理和优化中的常见操作

            float* P = sub_sparse_map->patch[i];
            for (int x=0; x<patch_size; x++) 
            {
                uint8_t* img_ptr = (uint8_t*) img.data + (v_ref_i+x*scale-patch_size_half*scale)*width + u_ref_i-patch_size_half*scale;
                for (int y=0; y<patch_size; ++y, img_ptr+=scale) 
                {
                    // if((level==2 && iteration==0) || (level==1 && iteration==0) || level==0)
                    //{
                    float du = 0.5f * ((w_ref_tl*img_ptr[scale] + w_ref_tr*img_ptr[scale*2] + w_ref_bl*img_ptr[scale*width+scale] + w_ref_br*img_ptr[scale*width+scale*2])
                                -(w_ref_tl*img_ptr[-scale] + w_ref_tr*img_ptr[0] + w_ref_bl*img_ptr[scale*width-scale] + w_ref_br*img_ptr[scale*width]));
                    float dv = 0.5f * ((w_ref_tl*img_ptr[scale*width] + w_ref_tr*img_ptr[scale+scale*width] + w_ref_bl*img_ptr[width*scale*2] + w_ref_br*img_ptr[width*scale*2+scale])
                                -(w_ref_tl*img_ptr[-scale*width] + w_ref_tr*img_ptr[-scale*width+scale] + w_ref_bl*img_ptr[0] + w_ref_br*img_ptr[scale]));
                    Jimg << du, dv; //1 * 2
                    Jimg = Jimg * (1.0/scale);
                    Jdphi = Jimg * Jdpi * p_hat; // 1* 3
                    Jdp = -Jimg * Jdpi; // 1 * 3
                    //Jdphi_dR R cam to imu
                    JdR = Jdphi * Jdphi_dR + Jdp * Jdp_dR; // 1 * 3
                    Jdt = Jdp * Jdp_dt; // 1 * 3
                    //}
                    //这里是真正算误差的地方，在图像平面算
                    double res = w_ref_tl*img_ptr[0] + w_ref_tr*img_ptr[scale] + w_ref_bl*img_ptr[scale*width] + w_ref_br*img_ptr[scale*width+scale]  - P[patch_size_total*level + x*patch_size+y];
                    z(i*patch_size_total+x*patch_size+y) = res;
                    // float weight = 1.0;
                    // if(iteration > 0)
                    //     weight = weight_function_->value(res/weight_scale_); 
                    // R(i*patch_size_total+x*patch_size+y) = weight;       
                    patch_error +=  res*res;
                    n_meas_++;
                    // H.block<1,6>(i*patch_size_total+x*patch_size+y,0) << JdR*weight, Jdt*weight;
                    // if((level==2 && iteration==0) || (level==1 && iteration==0) || level==0)
                    //H_sub 的行维度是点 * 64（8 * 8），列是6
                    H_sub.block<1,6>(i*patch_size_total+x*patch_size+y,0) << JdR, Jdt;
                }
            }  
            //每个点的patch误差
            sub_sparse_map->errors[i] = patch_error;
            //总误差
            error += patch_error;
        }

        // computeH += omp_get_wtime() - t1;

        //每一次迭代后
        error = error/n_meas_;

        // double t3 = omp_get_wtime();
        //如果新误差小于上一次误差，更新状态
        if (error <= last_error) 
        {
            old_state = (*state);
            last_error = error;

            // K = (H.transpose() / img_point_cov * H + state->cov.inverse()).inverse() * H.transpose() / img_point_cov;
            // auto vec = (*state_propagat) - (*state);
            // G = K*H;
            // (*state) += (-K*z + vec - G*vec);

            auto &&H_sub_T = H_sub.transpose();
            H_T_H.block<6,6>(0,0) = H_sub_T * H_sub;
            MD(DIM_STATE, DIM_STATE) &&K_1 = (H_T_H + (state->cov / img_point_cov).inverse()).inverse();
            auto &&HTz = H_sub_T * z;
            // K = K_1.block<DIM_STATE,6>(0,0) * H_sub_T;
            auto vec = (*state_propagat) - (*state);
            G.block<DIM_STATE,6>(0,0) = K_1.block<DIM_STATE,6>(0,0) * H_T_H.block<6,6>(0,0);
            auto solution = - K_1.block<DIM_STATE,6>(0,0) * HTz + vec - G.block<DIM_STATE,6>(0,0) * vec.block<6,1>(0,0);
            (*state) += solution;
            auto &&rot_add = solution.block<3,1>(0,0);
            auto &&t_add   = solution.block<3,1>(3,0);

            //小于某个值
            //这里57.3是180 / PI，所以乘以这个值就是把弧度换成角度，rot_add小于0.001度，
            if ((rot_add.norm() * 57.3f < 0.001f) && (t_add.norm() * 100.0f < 0.001f))
            {
                EKF_end = true;
            }
        }
        else //如果新误差大于上一次误差，直接结束
        {
            (*state) = old_state;
            EKF_end = true;
        }

        // ekf_time += omp_get_wtime() - t3;

        if (iteration==NUM_MAX_ITERATIONS || EKF_end) 
        {
            break;
        }
    }
    return last_error;
} 
//获取相机坐标系相对于世界坐标系的旋转矩阵和位移向量，这里IMU已经在点云去畸变的时候加入state了
void LidarSelector::updateFrameState(StatesGroup state)
{
    //此时body相对世界原点姿态
    M3D Rwi(state.rot_end);
    V3D Pwi(state.pos_end);
    Rcw = Rci * Rwi.transpose();
    Pcw = -Rci*Rwi.transpose()*Pwi + Pci;
    new_frame_->T_f_w_ = SE3(Rcw, Pcw);
}


//这里有40个像素判断，或者运动幅度大，加入地图，要加入pose。
void LidarSelector::addObservation(cv::Mat img)
{
    int total_points = sub_sparse_map->index.size();
    if (total_points==0) return;

    for (int i=0; i<total_points; i++) 
    {
        PointPtr pt = sub_sparse_map->voxel_points[i];
        if(pt==nullptr) continue;
        V2D pc(new_frame_->w2c(pt->pos_));
        SE3 pose_cur = new_frame_->T_f_w_;
        bool add_flag = false;
        // if (sub_sparse_map->errors[i]<= 100*patch_size_total && sub_sparse_map->errors[i]>0) //&& align_flag[i]==1) 
        {
            float* patch_temp = new float[patch_size_total*3];
            getpatch(img, pc, patch_temp, 0);
            getpatch(img, pc, patch_temp, 1);
            getpatch(img, pc, patch_temp, 2);

            //TODO: condition: distance and view_angle 
            // Step 1: time
            FeaturePtr last_feature =  pt->obs_.back();
            // if(new_frame_->id_ >= last_feature->id_ + 20) add_flag = true;

            // Step 2: delta_pose
            SE3 pose_ref = last_feature->T_f_w_;
            SE3 delta_pose = pose_ref * pose_cur.inverse();
            double delta_p = delta_pose.translation().norm();
            double delta_theta = (delta_pose.rotation_matrix().trace() > 3.0 - 1e-6) ? 0.0 : std::acos(0.5 * (delta_pose.rotation_matrix().trace() - 1));            
            if(delta_p > 0.5 || delta_theta > 10) add_flag = true;

            // Step 3: pixel distance
            Vector2d last_px = last_feature->px;
            double pixel_dist = (pc-last_px).norm();
            if(pixel_dist > 40) add_flag = true;
            
            // Maintain the size of 3D Point observation features.
            if(pt->obs_.size()>=20)
            {
                FeaturePtr ref_ftr;
                pt->getFurthestViewObs(new_frame_->pos(), ref_ftr);
                pt->deleteFeatureRef(ref_ftr);
                // ROS_WARN("ref_ftr->id_ is %d", ref_ftr->id_);
            } 
            if(add_flag)
            {
                pt->value = vk::shiTomasiScore(img, pc[0], pc[1]);
                Vector3d f = cam->cam2world(pc);
                FeaturePtr ftr_new(new Feature(patch_temp, pc, f, new_frame_->T_f_w_, pt->value, sub_sparse_map->search_levels[i])); 
                ftr_new->img = new_frame_->img_pyr_[0];
                ftr_new->id_ = new_frame_->id_;
                // ftr_new->ImgPyr.resize(5);
                // for(int i=0;i<5;i++) ftr_new->ImgPyr[i] = new_frame_->img_pyr_[i];
                //往sub_sparse_map里加Feature
                pt->addFrameRef(ftr_new);      
    
            }
        }
    }
}

void LidarSelector::ComputeJ(cv::Mat img) 
{
    //视觉子图,从第一步选出来
    int total_points = sub_sparse_map->index.size();
    if (total_points==0) return;
    float error = 1e10;
    float now_error = error;

    //三层金字塔
    for (int level=2; level>=0; level--) 
    {
        //
        now_error = UpdateState(img, error, level);
    }
    if (now_error < error)
    {
        state->cov -= G*state->cov;
    }
    //更新状态
    updateFrameState(*state);
}


void LidarSelector::display_visaul_select_point(double time){
    int total_points = add_voxel_points_.size();
    if(total_points == 0) return;
    for(int i = 0; i < total_points; i++) {
        V2D pc(new_frame_->w2c(add_voxel_points_[i]));
        // int index = static_cast<int>(pc[0]/grid_size)*grid_n_height + static_cast<int>(pc[1]/grid_size);
        if(visual_score[i] == 0) continue;
        cv::Point2f pf;
        pf = cv::Point2f(pc[0], pc[1]);   
        cv::circle(img_cp, pf, 4, cv::Scalar(0, 255, 0), -1, 8); // Green Sparse Align tracked
    }
    int lidar_points = display_lidar_points_.size();
    for(int i = 0; i < lidar_points; i++) {
        if(display_lidar_map_value[i] == 0) continue;
        cv::Point2f pf;
        pf = cv::Point2f(display_lidar_points_[i].x(), display_lidar_points_[i].y());
        cv::circle(img_cp, pf, 4, cv::Scalar(0, 0, 255), -1, 8); // Red Sparse Align tracked
    }
    std::string text = std::to_string(int(1/time))+" HZ";
    cv::Point2f origin;
    origin.x = 20;
    origin.y = 20;
    cv::putText(img_cp, text, origin, cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(255, 255, 255), 1, 8, 0);
}

void LidarSelector::display_keypatch(double time)
{
    int total_points = sub_sparse_map->index.size();
    if (total_points==0) return;
    for(int i=0; i<total_points; i++)
    {
        PointPtr pt = sub_sparse_map->voxel_points[i];
        V2D pc(new_frame_->w2c(pt->pos_));
        cv::Point2f pf;
        pf = cv::Point2f(pc[0], pc[1]); 
        if (sub_sparse_map->errors[i]<8000) // 5.5
            cv::circle(img_cp, pf, 4, cv::Scalar(0, 255, 0), -1, 8); // Green Sparse Align tracked
        else
            cv::circle(img_cp, pf, 4, cv::Scalar(255, 0, 0), -1, 8); // Blue Sparse Align tracked
    }   
    std::string text = std::to_string(int(1/time))+" HZ";
    cv::Point2f origin;
    origin.x = 20;
    origin.y = 20;
    cv::putText(img_cp, text, origin, cv::FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(255, 255, 255), 1, 8, 0);
}

V3F LidarSelector::getpixel(cv::Mat img, V2D pc) 
{
    const float u_ref = pc[0];
    const float v_ref = pc[1];
    const int u_ref_i = floorf(pc[0]); 
    const int v_ref_i = floorf(pc[1]);
    const float subpix_u_ref = (u_ref-u_ref_i);
    const float subpix_v_ref = (v_ref-v_ref_i);
    const float w_ref_tl = (1.0-subpix_u_ref) * (1.0-subpix_v_ref);
    const float w_ref_tr = subpix_u_ref * (1.0-subpix_v_ref);
    const float w_ref_bl = (1.0-subpix_u_ref) * subpix_v_ref;
    const float w_ref_br = subpix_u_ref * subpix_v_ref;
    uint8_t* img_ptr = (uint8_t*) img.data + ((v_ref_i)*width + (u_ref_i))*3;
    float B = w_ref_tl*img_ptr[0] + w_ref_tr*img_ptr[0+3] + w_ref_bl*img_ptr[width*3] + w_ref_br*img_ptr[width*3+0+3];
    float G = w_ref_tl*img_ptr[1] + w_ref_tr*img_ptr[1+3] + w_ref_bl*img_ptr[1+width*3] + w_ref_br*img_ptr[width*3+1+3];
    float R = w_ref_tl*img_ptr[2] + w_ref_tr*img_ptr[2+3] + w_ref_bl*img_ptr[2+width*3] + w_ref_br*img_ptr[width*3+2+3];
    V3F pixel(B,G,R);
    return pixel;
}

void LidarSelector::detect(cv::Mat img, PointCloudXYZI::Ptr pg) 
{
    if(width!=img.cols || height!=img.rows)
    {
        // std::cout<<"Resize the img scale !!!"<<std::endl;
        double scale = 0.5;
        cv::resize(img,img,cv::Size(img.cols*scale,img.rows*scale),0,0,CV_INTER_LINEAR);
    }
    img_rgb = img.clone();
    img_cp = img.clone();
    cv::cvtColor(img,img,CV_BGR2GRAY);

    //创建一个新的frame,然后Img作为金字塔第一层
    new_frame_.reset(new Frame(cam, img.clone()));
    //只是获取相机到世界坐标系的变换，设置一下new_frame
    updateFrameState(*state);

    //第一次 pg-点云
    if(stage_ == STAGE_FIRST_FRAME && pg->size()>10)
    {
        //关键帧
        new_frame_->setKeyframe();
        stage_ = STAGE_DEFAULT_FRAME;
    }

    double t1 = omp_get_wtime();

    //利用pg最近雷达扫描点，获取视觉子图
    addFromSparseMap(img, pg);

    double t3 = omp_get_wtime();

    addSparseMap(img, pg);

    double t4 = omp_get_wtime();
    
    // computeH = ekf_time = 0.0;
    //更新状态
    ComputeJ(img);

    double t5 = omp_get_wtime();
    //用视觉点更新
    addObservation(img);
    
    double t2 = omp_get_wtime();
    
    frame_cont ++;
    ave_total = ave_total * (frame_cont - 1) / frame_cont + (t2 - t1) / frame_cont;

    printf("[ VIO ]: time: addFromSparseMap: %0.6f addSparseMap: %0.6f ComputeJ: %0.6f addObservation: %0.6f total time: %0.6f ave_total: %0.6f.\n"
    , t3-t1, t4-t3, t5-t4, t2-t5, t2-t1);

    display_visaul_select_point(t2 - t1);
    // display_keypatch(t2-t1);
} 

} // namespace lidar_selection