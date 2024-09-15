// #include <iostream>
// #include <vector>
// #include <algorithm>
// #include <cmath>
// #include <numeric>
// #include <tuple>
// #include <random>
// #include "FlockingEnv.h"

// // 用于生成随机数的函数
// double generateRandomNumber() {
//     static std::mt19937 gen(std::random_device{}());
//     static std::uniform_real_distribution<double> dis(0.0, 1.0);
//     return dis(gen);
// }

// // 初始化函数
// Env initializeEnv(int rows, int cols, int leader_state_size) {
//     Env env;
//     // 初始化 p 为 2 行 10 列的随机数组
//     env.p.resize(2, std::vector<double>(10));
//     for (int i = 0; i < 2; ++i) {
//         for (int j = 0; j < 10; ++j) {
//             env.p[i][j] = generateRandomNumber();
//         }
//     }

//     // 初始化 dp 为相同大小的零矩阵
//     env.dp.resize(rows, std::vector<double>(cols, 0.0));

//     // 初始化 obs 为相同形状的零矩阵
//     env.obs.resize(16, std::vector<double>(cols, 0.0));
    
//     // 初始化 leader_state 为零元素数组
//     env.leader_state.resize(leader_state_size, 0.0);

//     // 其他属性的初始化
//     env.d_sen = 8;
//     env.topo_nei_max = 2;
//     env.n_a = cols;
//     env.obs_dim_agent = 16;
//     env.dim = 2;
//     // 可以根据实际情况设置其他属性的值

//     return env;
// }

// int main() {
//     // 使用示例
//     Env myEnv = initializeEnv(2, 10, 4);
//     // 输出示例
//     std::cout << "p matrix:" << std::endl;
//     for (const auto& row : myEnv.p) {
//         for (const auto& val : row) {
//             std::cout << val << " ";
//         }
//         std::cout << std::endl;
//     }
//     std::cout << "obs matrix:" << std::endl;
//     for (const auto& row : myEnv.obs) {
//         for (const auto& val : row) {
//             std::cout << val << " ";
//         }
//         std::cout << std::endl;
//     }
//     std::cout << "leader_state vector:" << std::endl;
//     for (const auto& val : myEnv.leader_state) {
//         std::cout << val << " ";
//     }
//     std::cout << std::endl;


//     // _get_observation(myEnv);

//     std::cout << "obs matrix:" << std::endl;
//     for (const auto& row : myEnv.obs) {
//         for (const auto& val : row) {
//             std::cout << val << " ";
//         }
//         std::cout << std::endl;
//     }


//     return 0;
// }

// #include <iostream>
// #include <vector>
// #include <cmath>
// #include <algorithm>
// #include <numeric>
// #include "FlockingEnv.h"


// // Define a function to calculate norm of a vector
// double norm(const std::vector<double>& v) {
//     double sum = 0.0;
//     for (double x : v) {
//         sum += x * x;
//     }
//     return std::sqrt(sum);
// }

// // Define a function to calculate cosine decay
// double rho_cos_dec(double dist, double dc, double dr, double lambda) {
//     if (dist < dc) {
//         return 1.0;
//     } else if (dist >= dr) {
//         return 0.0;
//     } else {
//         return std::pow(std::cos(M_PI * (dist - dc) / (2 * (dr - dc))), lambda);
//     }
// }

// int main() {
//     // Example data
//     int n_a = 3; // Number of agents
//     int n_ao = 6; // Number of agents + obstacles
//     int topo_nei_max = 2; // Maximum number of neighbors
//     double d_sen = 5; // Sensor range
//     double d_ref = 0.3; // Reference distance

//     // Example agent positions
//     std::vector<std::vector<double>> p = {{0.0, 1.0, 2.0}, {0.0, 1.0, 2.0}};
//     std::vector<std::vector<double>> dp = {{0.1, 0.1, 0.2}, {0.0, 0.1, 1.0}};
//     std::vector<std::vector<double>> act = {{1.0, 1.0, 2.0}, {0.0, 1.0, 2.0}};

//     // Calculate distance matrix
//     std::vector<std::vector<double>> dist_mat(n_a, std::vector<double>(n_a, 0.0));
//     for (int i = 0; i < n_a; ++i) {
//         for (int j = 0; j < n_a; ++j) {
//             dist_mat[i][j] = norm({p[0][j] - p[0][i], p[1][j] - p[1][i]});
//         }
//     }

//     // Sort indices based on distance matrix
//     std::vector<std::vector<int>> sorted_indices(n_a, std::vector<int>(n_a, 0));
//     for (int i = 0; i < n_a; ++i) {
//         std::iota(sorted_indices[i].begin(), sorted_indices[i].end(), 0);
//         std::sort(sorted_indices[i].begin(), sorted_indices[i].end(), [&](int a, int b) { 
//             return dist_mat[a][i] < dist_mat[b][i]; 
//         });
//     }

//     // Calculate B matrix
//     std::vector<std::vector<double>> B(n_a, std::vector<double>(n_a, 0.0));
//     for (int i = 0; i < n_a; ++i) {
//         for (int j = 0; j < n_a; ++j) {
//             B[j][i] = dist_mat[sorted_indices[i][j]][i];
//         }
//     }

//     // Calculate I matrix
//     std::vector<std::vector<int>> I(n_a, std::vector<int>(n_a, 0));
//     for (int i = 0; i < n_a; ++i) {
//         for (int j = 0; j < n_a; ++j) {
//             I[j][i] = sorted_indices[i][j];
//         }
//     }

//     // Initialize reward_a matrix
//     std::vector<double> reward_a(n_ao, 0.0);

//     // Inter-agent reward
//     bool penalize_distance = true;
//     if (penalize_distance) {
//         for (int agent = 0; agent < n_a; ++agent) {
//             std::vector<int> list_nei_indices;
//             for (int j = 0; j < n_a; ++j) {
//                 if (B[j][agent] <= d_sen) {
//                     list_nei_indices.push_back(I[j][agent]);
//                 }
//             }
            
//             std::vector<int> list_nei;
//             for (int i = 0; i < list_nei_indices.size(); ++i) {
//                 if (i > 0 && i < topo_nei_max + 1) {
//                     list_nei.push_back(list_nei_indices[i]);
//                 }
//             }

//             std::vector<double> pos_rel(2, 0.0);
//             std::vector<double> avg_neigh_vel(2, 0.0);

//             if (!list_nei.empty()) {
//                 for (int agent2 : list_nei) {
//                     pos_rel = {p[0][agent2] - p[0][agent], p[1][agent2] - p[1][agent]};
//                     // reward_a[agent] -= 8 * rho_cos_dec(norm(pos_rel), 0.1, d_ref, 1);
//                     // reward_a[agent] -= 0.7 * (0.4 * rho_cos_dec(norm(pos_rel), 0.1, d_ref, 1) + 1) * norm(_extract_column_one(act, agent));
//                     reward_a[agent] -= 0.3 / (norm(pos_rel) + 0.001);
//                     reward_a[agent] -= 0.3 * norm(pos_rel);
//                     avg_neigh_vel[0] += dp[0][agent2] / norm(_extract_column_one(dp, agent2));
//                     avg_neigh_vel[1] += dp[1][agent2] / norm(_extract_column_one(dp, agent2));
//                 }
                    
//                 avg_neigh_vel[0] /= list_nei.size();
//                 avg_neigh_vel[1] /= list_nei.size();

//                 double norm_dp_agent = norm(_extract_column_one(dp, agent));
//                 double vel_diff_norm = sqrt(pow(avg_neigh_vel[0] - dp[0][agent] / norm_dp_agent, 2) +
//                                         pow(avg_neigh_vel[1] - dp[1][agent] / norm_dp_agent, 2));
//                 reward_a[agent] -= 4 * vel_diff_norm;

//             }
//         }
//     }

//     // Other reward calculations go here...

//     // Output reward_a matrix
//     std::cout << "Reward_a matrix:" << std::endl;
//     for (double value : reward_a) {
//         std::cout << value << " ";
//     }
//     std::cout << std::endl;

//     return 0;
// }


// #include <iostream>
// #include <vector>
// #include <utility>

// // 将 _get_dist_b2w 函数放在这里

// std::pair<std::vector<std::vector<float>>, std::vector<std::vector<bool>>> get_dist_b2w(const std::vector<std::vector<float>>& p, const std::vector<float>& r, float boundary_L_half) {
//     int n_ao = p[0].size();
//     std::vector<std::vector<float>> d_b2w(4, std::vector<float>(n_ao, 0.0));
//     std::vector<std::vector<bool>> isCollision(4, std::vector<bool>(n_ao, false));
    
//     for (int i = 0; i < n_ao; ++i) {
//         d_b2w[0][i] = p[0][i] - r[i] - (-boundary_L_half);
//         d_b2w[1][i] = boundary_L_half - (p[1][i] + r[i]);
//         d_b2w[2][i] = boundary_L_half - (p[0][i] + r[i]);
//         d_b2w[3][i] = p[1][i] - r[i] - (-boundary_L_half);
//     }
    
//     for (int i = 0; i < 4; ++i) {
//         for (int j = 0; j < n_ao; ++j) {
//             isCollision[i][j] = (d_b2w[i][j] < 0);
//             d_b2w[i][j] = std::abs(d_b2w[i][j]);
//         }
//     }
    
//     return std::make_pair(d_b2w, isCollision);
// }

// int main() {
//     // 代理的位置 p 和半径 r
//     std::vector<std::vector<float>> p = {{0.0, 1.0, 0.0}, {1.0, 1.0, 2.0}};
//     std::vector<float> r = {0.5, 0.6, 0.7};
    
//     // 边界的一半长度
//     float boundary_L_half = 5.0;
    
//     // 调用 _get_dist_b2w 函数
//     auto result = get_dist_b2w(p, r, boundary_L_half);
    
//     // 打印距离数组
//     std::cout << "Distance array:" << std::endl;
//     for (const auto& row : result.first) {
//         for (float dist : row) {
//             std::cout << dist << " ";
//         }
//         std::cout << std::endl;
//     }
    
//     // 打印是否碰撞数组
//     std::cout << "Collision array:" << std::endl;
//     for (const auto& row : result.second) {
//         for (bool collide : row) {
//             std::cout << collide << " ";
//         }
//         std::cout << std::endl;
//     }
    
//     return 0;
// }

#include <iostream>
#include <typeinfo>

int main() {
    int i = 10;
    float f = 3.14;
    double d = 2.718;
    char c = 'A';
    bool b = true;

    std::cout << "Type of i: " << typeid(i).name() << std::endl;
    std::cout << "Type of f: " << typeid(f).name() << std::endl;
    std::cout << "Type of d: " << typeid(d).name() << std::endl;
    std::cout << "Type of c: " << typeid(c).name() << std::endl;
    std::cout << "Type of b: " << typeid(b).name() << std::endl;

    return 0;
}



