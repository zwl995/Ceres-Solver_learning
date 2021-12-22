#if 0
//ʹ����򵥵ĺ������Ӳ���Ceres-Solver
#include <iostream>
#include <ceres/ceres.h>

using namespace std;
using namespace ceres;
//��һ���֣��������ۺ���
struct CostFunctor {
    template <typename T>
    bool operator()(const T* const x, T* residual) const {
        residual[0] = T(10.0) - x[0];
        return true;
    }
};

//������
int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);

    // Ѱ�Ų���x�ĳ�ʼֵ��Ϊ5
    double initial_x = 0.5;
    double x = initial_x;

    // �ڶ����֣�����Ѱ������
    Problem problem;
    //ʹ���Զ��󵼣���֮ǰ�Ĵ��ۺ����ṹ�崫�룬��һ��1�����ά�ȣ�
    //���в��ά�ȣ��ڶ���1������ά�ȣ�����Ѱ�Ų���x��ά�ȡ�
    CostFunction* cost_function =
        new AutoDiffCostFunction<CostFunctor, 1, 1>(new CostFunctor); 
    problem.AddResidualBlock(cost_function, nullptr, &x); //���������������������Ƚϼ򵥣����һ�����С�

    //�������֣� ���ò����������
    Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR; //�����������̵Ľⷨ
    options.minimizer_progress_to_stdout = true;//�����cout
    Solver::Summary summary;//�Ż���Ϣ
    Solve(options, &problem, &summary);//���!!!

    std::cout << summary.BriefReport() << "\n";//����Ż��ļ�Ҫ��Ϣ
    //���ս��
    std::cout << "x : " << initial_x
        << " -> " << x << "\n";
    return 0;
}
#endif 