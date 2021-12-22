#if 0
//ʹ����΢���ӵĶ�Լ��������������Ceres-Solver
#include <iostream>
#include <ceres/ceres.h>

using namespace std;
using namespace ceres;
 
//��һ���֣������ĸ����ۺ���
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

	//��ʼ��Ѱ�Ų�����x1,x2,x3,x4
	double x1 = 3.0, x2 = -1.0, x3 = 0.0, x4 = 1.0;
	
	//�ڶ����֣�����Ѱ������
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

	//�������֣����ò����������
	Solver::Options options;
	options.linear_solver_type = ceres::DENSE_QR;	//�����������̵Ľⷨ
	options.minimizer_progress_to_stdout = true;	//�����cout
	Solver::Summary summary;						//�Ż���Ϣ
	Solve(options, &problem, &summary);			//���

	std::cout << summary.BriefReport() << std::endl;//����Ż��ļ�Ҫ��Ϣ
	//��ӡ���
	std::cout << "x1:" << x1 << std::endl;
	std::cout << "x2:" << x2 << std::endl;
	std::cout << "x3:" << x3 << std::endl;
	std::cout << "x4:" << x4 << std::endl;
}
#endif 