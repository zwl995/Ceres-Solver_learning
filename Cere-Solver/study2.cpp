#if 0
//使用稍微复杂的多约束条件函数测试Ceres-Solver
#include <iostream>
#include <ceres/ceres.h>

using namespace std;
using namespace ceres;
 
//第一部分：构造四个代价函数
struct F1 {
	template<typename T>
	bool operator()(const T* const x1, const T* const x2, T* residual) const {
		residual[0] = x1[0] + 10.0 * x2[0];
		return true;
	}
};

struct F2 {
	template<typename T>
	bool operator()(const T* const x3, const T* const x4, T* residual) const {
		residual[0] = sqrt(5.0) * (x3[0] - x4[0]);
		return true;
	}
};

struct F3 {
	template<typename T>
	bool operator()(const T* const x2, const T* const x3, T* residual) const {
		residual[0] = (x2[0] - 2.0 * x3[0]) * (x2[0] - 2.0 * x3[0]);
		return true;
	}
};

struct F4 {
	template<typename T>
	bool operator()(const T* const x1, const T* const x4, T* residual) const {
		residual[0] = sqrt(10.0) * (x1[0] - x4[0]) * (x1[0] - x4[0]);
		return true;
	}
};

int main(int argc, char** argv) {

	//初始化寻优参数：x1,x2,x3,x4
	double x1 = 3.0, x2 = -1.0, x3 = 0.0, x4 = 1.0;
	
	//第二部分：构建寻优问题
	Problem problem;
	problem.AddResidualBlock(
		new AutoDiffCostFunction<F1, 1, 1, 1>(new F1), nullptr, &x1, &x2
	);
	problem.AddResidualBlock(
		new AutoDiffCostFunction<F2, 1, 1, 1>(new F2), nullptr, &x3, &x4
	);
	problem.AddResidualBlock(
		new AutoDiffCostFunction<F3, 1, 1, 1>(new F3), nullptr, &x2, &x3
	);
	problem.AddResidualBlock(
		new AutoDiffCostFunction<F4, 1, 1, 1>(new F4), nullptr, &x1, &x4
	);

	//第三部分：配置并运行求解器
	Solver::Options options;
	options.linear_solver_type = ceres::DENSE_QR;	//配置增量方程的解法
	options.minimizer_progress_to_stdout = true;	//输出到cout
	Solver::Summary summary;						//优化信息
	Solve(options, &problem, &summary);			//求解

	std::cout << summary.BriefReport() << std::endl;//输出优化的简要信息
	//打印结果
	std::cout << "x1:" << x1 << std::endl;
	std::cout << "x2:" << x2 << std::endl;
	std::cout << "x3:" << x3 << std::endl;
	std::cout << "x4:" << x4 << std::endl;
}
#endif 