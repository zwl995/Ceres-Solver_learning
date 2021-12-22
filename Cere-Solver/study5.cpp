#if 1
//使用BAL数据集一个数据测试BA
#include <cmath>
#include <cstdio>
#include <iostream>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

using namespace std;
using namespace ceres;

//写一个读取BAL数据集的类，使用指针的方式来读取与存放数据，值得学习
class BALProblem {
public:
    ~BALProblem() { //析构函数，将私有类中的指针清理掉
        delete[] point_index_;
        delete[] camera_index_;
        delete[] observations_;
        delete[] parameters_;
    }
    int num_observations() const { return num_observations_; }
    const double* observations() const { return observations_; }
    double* mutable_cameras() { return parameters_; }
    double* mutable_points() { return parameters_ + 9 * num_cameras_; }
    double* mutable_camera_for_observation(int i) {
        return mutable_cameras() + camera_index_[i] * 9;
    }
    double* mutable_point_for_observation(int i) {
        return mutable_points() + point_index_[i] * 3;
    }
    bool LoadFile(const char* filename) {
        FILE* fptr;
        if (fopen_s(&fptr, filename, "r")) {
            return false;
        };
        FscanfOrDie(fptr, "%d", &num_cameras_);
        FscanfOrDie(fptr, "%d", &num_points_);
        FscanfOrDie(fptr, "%d", &num_observations_);
        point_index_ = new int[num_observations_];
        camera_index_ = new int[num_observations_];
        observations_ = new double[2 * num_observations_];
        num_parameters_ = 9 * num_cameras_ + 3 * num_points_;
        parameters_ = new double[num_parameters_];
        for (int i = 0; i < num_observations_; ++i) {
            FscanfOrDie(fptr, "%d", camera_index_ + i);
            FscanfOrDie(fptr, "%d", point_index_ + i);
            for (int j = 0; j < 2; ++j) {
                FscanfOrDie(fptr, "%lf", observations_ + 2 * i + j);
            }
        }
        for (int i = 0; i < num_parameters_; ++i) {
            FscanfOrDie(fptr, "%lf", parameters_ + i);
        }
        return true;
    }
private:
    template <typename T>
    void FscanfOrDie(FILE* fptr, const char* format, T* value) {
        int num_scanned = fscanf_s(fptr, format, value);
        if (num_scanned != 1) {
            LOG(FATAL) << "Invalid UW data file.";
        }
    }
    int num_cameras_;   //相机数量
    int num_points_;    //三维点数量
    int num_observations_;  //观察到的2D像素坐标数量
    int num_parameters_;    //参数数量：9*相机数量+3*点数量
    int* point_index_;  //通过指针的形式来存放点
    int* camera_index_; //通过指针的形式来存放相机
    double* observations_;  //通过指针的形式来存放观察到的2D像素坐标
    double* parameters_;    //通过指针的形式来存放参数
};

//第一部分：构造代价函数
struct SnavelyReprojectionError {
    SnavelyReprojectionError(double observed_x, double observed_y)
        : observed_x(observed_x), observed_y(observed_y) {}
    template <typename T>
    bool operator()(const T* const camera,
        const T* const point,
        T* residuals) const {

        //==========先旋转，再平移得到3D坐标==========
        T p[3]; // 3D点坐标
        // camera[0,1,2] are the angle-axis rotation.
        ceres::AngleAxisRotatePoint(camera, point, p);
        // camera[3,4,5] are the translation.
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];
        //==========先旋转，再平移得到3D坐标==========


        //==========根据投影关系与相机畸变模型得到投影后的预测像素坐标值==========
        // Compute the center of distortion. The sign change comes from
        // the camera model that Noah Snavely's Bundler assumes, whereby
        // the camera coordinate system has a negative z axis.
        T xp = -p[0] / p[2];
        T yp = -p[1] / p[2];
        // Apply second and fourth order radial distortion.
        const T& l1 = camera[7];
        const T& l2 = camera[8];
        T r2 = xp * xp + yp * yp;
        T distortion = 1.0 + r2 * (l1 + l2 * r2);
        // Compute final projected point position.
        const T& focal = camera[6];
        T predicted_x = focal * distortion * xp;
        T predicted_y = focal * distortion * yp;
        //==========根据投影关系与相机畸变模型得到投影后的预测像素坐标值==========

        // The error is the difference between the predicted and observed position.
        residuals[0] = predicted_x - observed_x;
        residuals[1] = predicted_y - observed_y;
        return true;
    }
    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create(const double observed_x,
        const double observed_y) {
        return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>(
            new SnavelyReprojectionError(observed_x, observed_y)));
    }
    double observed_x;
    double observed_y;
};

int main(int argc, char** argv) {
    BALProblem bal_problem;
    if (!bal_problem.LoadFile("../data/problem-49-7776-pre.txt")) {
        std::cout<<"Load File fail！"<<std::endl;
    }
    //从数据集中读取观测的像素值
    const double* observations = bal_problem.observations();

    //第二部分：构建寻优问题
    ceres::Problem problem;
    for (int i = 0; i < bal_problem.num_observations(); ++i) {
        ceres::CostFunction* cost_function = SnavelyReprojectionError::Create(
            observations[2 * i + 0], observations[2 * i + 1]);
        problem.AddResidualBlock(cost_function,
            NULL /* squared loss */,
            //初始化寻优参数放到函数内部了
            bal_problem.mutable_camera_for_observation(i),
            bal_problem.mutable_point_for_observation(i));
    }

    //第三部分：配置并运行求解器
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;    //此处求解器不一样
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";
    return 0;
}

#endif 