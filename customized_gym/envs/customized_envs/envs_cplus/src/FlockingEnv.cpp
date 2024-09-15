#include "FlockingEnv.h"
#include <iostream>
#include <tuple>
#include <algorithm>
#include <vector>
#include <cmath> 
#include <numeric>
#include <typeinfo>

// Function to calculate observation for each env
void _get_observation(double *p_input, 
                      double *dp_input, 
                      double *heading_input,
                      double *obs_input, 
                      double *leader_state_input, 
                      int *neighbor_index_input, 
                      int *leader_index_input,
                      int *random_permutation,
                      double *boundary_pos_input, 
                      double d_sen, 
                      int topo_nei_max, 
                      int n_a, 
                      int n_l,
                      int obs_dim_agent, 
                      int dim, 
                      bool *condition) 
{
    Matrix matrix_p(dim, std::vector<double>(n_a));
    Matrix matrix_dp(dim, std::vector<double>(n_a));
    Matrix matrix_heading(dim, std::vector<double>(n_a));
    Matrix matrix_leader_state(2 * dim, std::vector<double>(n_l));
    Matrix matrix_obs(obs_dim_agent, std::vector<double>(n_a));
    std::vector<std::vector<int>> neighbor_index(n_a, std::vector<int>(topo_nei_max, -1));
    std::vector<double> boundary_pos(4, 0.0);

    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < n_a; ++j) {
            matrix_p[i][j] = p_input[i * n_a + j];
            matrix_dp[i][j] = dp_input[i * n_a + j];
            matrix_heading[i][j] = heading_input[i * n_a + j];
            // std::cout << matrix_p[i][j] << " ";
        }
        // std::cout << std::endl;
    }

    for (int i = 0; i < 2 * dim; ++i) {
        for (int j = 0; j < n_l; ++j) {
            matrix_leader_state[i][j] = leader_state_input[i * n_l + j];
        }
    }

    for (int i = 0; i < 4; ++i) {
        boundary_pos[i] = boundary_pos_input[i];
    }

    double L = (boundary_pos[2] - boundary_pos[0]) / 2.0;
    std::vector<double> shill_agent_state(4, 0.0);
    for (int agent_i = 0; agent_i < n_a; ++agent_i) {
        // Calculate relative positions and velocities
        Matrix relPos_a2a(dim, std::vector<double>(n_a, 0.0));
        Matrix relVel_a2a(dim, std::vector<double>(n_a, 0.0));

        for (int j = 0; j < n_a; ++j) {
            for (int k = 0; k < dim; ++k) {
                relPos_a2a[k][j] = matrix_p[k][j] - matrix_p[k][agent_i];
                if (condition[3]) {
                    relVel_a2a[k][j] = matrix_dp[k][j] - matrix_dp[k][agent_i];
                } else {
                    relVel_a2a[k][j] = matrix_heading[k][j] - matrix_heading[k][agent_i];
                }
            }
        }

        if (condition[0]) {
            _make_periodic(relPos_a2a, L, boundary_pos, true);
            // std::cout << L << std::endl;
        }

        // Obtain focused observations
        std::tuple<Matrix, Matrix, std::vector<int>> focused_obs = _get_focused(relPos_a2a, relVel_a2a, d_sen, topo_nei_max, true);
        Matrix relPos_a2a_focused = std::get<0>(focused_obs);
        Matrix relVel_a2a_focused = std::get<1>(focused_obs);
        std::vector<int> nei_index = std::get<2>(focused_obs);
        
        for (int i = 0; i < nei_index.size(); ++i) {
            neighbor_index[agent_i][i] = nei_index[i];
        }

        Matrix relPos_a2a_permutated = relPos_a2a_focused;
        Matrix relVel_a2a_permutated = relVel_a2a_focused;
        if (condition[4]) { // whether permutate the order of neighbors
            for (size_t i = 0; i < relPos_a2a_focused.size(); ++i) {
                for (size_t j = 0; j < topo_nei_max; ++j) {
                    relPos_a2a_permutated[i][j] = relPos_a2a_focused[i][random_permutation[j]];
                    relVel_a2a_permutated[i][j] = relVel_a2a_focused[i][random_permutation[j]];
                }
            }
        }

        Matrix obs_agent;
        if (condition[5]) { // whether contain myself state in the observation
            Matrix obs_agent_pos = _concatenate(_extract_column(matrix_p, agent_i), relPos_a2a_permutated, 1);
            Matrix obs_agent_vel = _concatenate(_extract_column(matrix_dp, agent_i), relVel_a2a_permutated, 1);
            obs_agent = _concatenate(obs_agent_pos, obs_agent_vel, 0);
        } else {
            obs_agent = _concatenate(relPos_a2a_permutated, relVel_a2a_permutated, 0);
        }

        // 将 obs_agent 转置，并展平为一维数组，然后赋值给 obs 的前部分
        std::vector<double> obs_agent_flat;
        obs_agent_flat.reserve(obs_agent.size() * obs_agent[0].size());
        for (size_t j = 0; j < obs_agent[0].size(); ++j) {
            for (size_t i = 0; i < obs_agent.size(); ++i) {
                obs_agent_flat.push_back(obs_agent[i][j]);
            }
        }

        if (condition[1] && condition[2]) { // is_leader == true
            // the position and velocity of nearest leader
            std::vector<double> leader_pos_rel_flat(dim, 0.0);
            std::vector<double> leader_vel_rel_flat(dim, 0.0);
            std::vector<double> leader_heading_rel_flat(dim, 0.0);

            // position and velocity of virtual leader
            Matrix leader_pos_rel_mat(dim, std::vector<double>(n_l));
            for (int j = 0; j < n_l; ++j) {
                for (int k = 0; k < dim; ++k) {
                    leader_pos_rel_mat[k][j] = matrix_leader_state[k][j] - matrix_p[k][agent_i];
                }
            }

            if (condition[0]) {
                _make_periodic(leader_pos_rel_mat, L, boundary_pos, true);
            }

            int min_index = 0;
            if (n_l > 1) {
                // 计算每列的模值
                std::vector<double> leader_agent_norms;
                for (size_t j = 0; j < leader_pos_rel_mat[0].size(); ++j) {
                    double norm_value = 0.0;
                    for (size_t i = 0; i < leader_pos_rel_mat.size(); ++i) {
                        norm_value += std::pow(leader_pos_rel_mat[i][j], 2);
                    }
                    norm_value = std::sqrt(norm_value);
                    leader_agent_norms.push_back(norm_value);
                }

                // 找出模值最小的列的索引
                auto min_it = std::min_element(leader_agent_norms.begin(), leader_agent_norms.end());
                min_index = std::distance(leader_agent_norms.begin(), min_it);
            }

            std::vector<double> leader_vel(dim, 0.0);
            for (int j = 0; j < dim; ++j) {
                leader_pos_rel_flat[j] = 0.0;
                leader_vel_rel_flat[j] = 0.0;
                leader_heading_rel_flat[j] = 0.0;
            }

            double norm_value_ = 0.0;
            for (size_t i = 0; i < leader_pos_rel_mat.size(); ++i) {
                norm_value_ += std::pow(leader_pos_rel_mat[i][min_index], 2);
            }
            norm_value_ = std::sqrt(norm_value_);
            if (norm_value_ < d_sen) {
                // output the index of the nearest leader
                leader_index_input[agent_i] = min_index;
                for (int j = 0; j < dim; ++j) {
                    leader_pos_rel_flat[j] = leader_pos_rel_mat[j][min_index];
                    leader_vel_rel_flat[j] = matrix_leader_state[dim + j][min_index] - matrix_dp[j][agent_i];
                }
                // relative heading
                for (size_t j = 0; j < dim; ++j) {
                    leader_vel[j] = matrix_leader_state[dim + j][min_index];
                }
                double leader_vel_norm = _norm(leader_vel) + 1E-8;
                for (int j = 0; j < dim; ++j) {
                    leader_heading_rel_flat[j] = (matrix_leader_state[dim + j][min_index] / leader_vel_norm - matrix_heading[j][agent_i]);
                }
            }

            if (condition[3]) { // dynamics_mode == Cartesian
                for (int j = 0; j < obs_dim_agent - 2 * dim; ++j) {
                    matrix_obs[j][agent_i] = obs_agent_flat[j];
                }
                for (int j = 0; j < dim; ++j) {
                    matrix_obs[obs_dim_agent - 2 * dim + j][agent_i] = leader_pos_rel_flat[j];
                }
                for (int j = 0; j < dim; ++j) {
                    matrix_obs[obs_dim_agent - dim + j][agent_i] = leader_vel_rel_flat[j];
                }
            } else { // dynamics_mode == Polar
                for (int j = 0; j < obs_dim_agent - 3 * dim; ++j) {
                    matrix_obs[j][agent_i] = obs_agent_flat[j];
                }
                for (int j = 0; j < dim; ++j) {
                    matrix_obs[obs_dim_agent - 3 * dim + j][agent_i] = leader_pos_rel_flat[j];
                }
                for (int j = 0; j < dim; ++j) {
                    matrix_obs[obs_dim_agent - 2 * dim + j][agent_i] = leader_heading_rel_flat[j];
                }
                for (int j = 0; j < dim; ++j) {
                    matrix_obs[obs_dim_agent - dim + j][agent_i] = matrix_heading[j][agent_i];
                }
            }
        } else { // is_leader == false
            if (condition[3]) { //dynamics_mode == Cartesian
                for (int j = 0; j < obs_dim_agent; ++j) {
                    matrix_obs[j][agent_i] = obs_agent_flat[j];
                }
            } else { //dynamics_mode == Polar
                for (int j = 0; j < obs_dim_agent - dim; ++j) {
                    matrix_obs[j][agent_i] = obs_agent_flat[j];
                }
                for (int j = 0; j < dim; ++j) {
                    matrix_obs[obs_dim_agent - dim + j][agent_i] = matrix_heading[j][agent_i];
                }

            }
        }

    }

    for (int i = 0; i < obs_dim_agent; ++i) {
        for (int j = 0; j < n_a; ++j) {
            obs_input[i * n_a + j] = matrix_obs[i][j];
        }
    }

    for (int i = 0; i < n_a; ++i) {
        for (int j = 0; j < topo_nei_max; ++j) {
            neighbor_index_input[i * topo_nei_max + j] = neighbor_index[i][j];
            // std::cout << neighbor_index[i][j] << "\t";
        }
    }

}

void _get_reward(double *p_input, 
                 double *dp_input,
                 double *heading_input, 
                 double *act_input, 
                 double *reward_input, 
                 double *leader_state_input, 
                 int *neighbor_index_input,
                 int *leader_index_input,
                 double *boundary_pos_input, 
                 double d_sen, 
                 double d_ref, 
                 int topo_nei_max, 
                 int n_a, 
                 int n_l,
                 int dim, 
                 bool *condition, 
                 bool *is_collide_b2b_input, 
                 bool *is_collide_b2w_input, 
                 double *coefficients) 
{
    // std::cout <<  d_sen << typeid(d_sen).name() << ":" << d_ref << typeid(d_ref).name() << std::endl;
    
    Matrix matrix_p(dim, std::vector<double>(n_a));
    Matrix matrix_dp(dim, std::vector<double>(n_a));
    Matrix matrix_heading(dim, std::vector<double>(n_a));
    Matrix matrix_act(dim, std::vector<double>(n_a));
    Matrix matrix_leader_state(2 * dim, std::vector<double>(n_l));
    std::vector<std::vector<int>> neighbor_index(n_a, std::vector<int>(topo_nei_max, -1));
    // std::vector<std::vector<int>> neighbor_index_last(n_a, std::vector<int>(topo_nei_max, -1));
    std::vector<double> boundary_pos(4, 0.0);
    std::vector<std::vector<bool>> is_collide_b2b(n_a, std::vector<bool>(n_a, false));
    std::vector<std::vector<bool>> is_collide_b2w(4, std::vector<bool>(n_a, false));

    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < n_a; ++j) {
            matrix_p[i][j] = p_input[i * n_a + j];
            matrix_dp[i][j] = dp_input[i * n_a + j];
            matrix_heading[i][j] = heading_input[i * n_a + j];
        }
    }

    for (int j = 0; j < n_a; ++j) {
        for (int i = 0; i < dim; ++i) {
            matrix_act[i][j] = act_input[j * dim + i];
            // std::cout << act_input[j * dim + i] << " ";
        }
    }

    for (int i = 0; i < 2 * dim; ++i) {
        for (int j = 0; j < n_l; ++j) {
            matrix_leader_state[i][j] = leader_state_input[i * n_l + j];
        }
    }

    for (int i = 0; i < n_a; ++i) {
        for (int j = 0; j < topo_nei_max; ++j) {
            neighbor_index[i][j] = neighbor_index_input[i * topo_nei_max + j];
            // neighbor_index_last[i][j] = neighbor_index_last_input[i * topo_nei_max + j];
            // std::cout << neighbor_index[i][j] << " ";
        }
        // std::cout << std::endl;
    }

    for (int i = 0; i < 4; ++i) {
        boundary_pos[i] = boundary_pos_input[i];
    }

    for (int i = 0; i < n_a; ++i) {
        for (int j = 0; j < n_a; ++j) {
            is_collide_b2b[i][j] = is_collide_b2b_input[i * n_a + j];
        }
    }
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < n_a; ++j) {
            is_collide_b2w[i][j] = is_collide_b2w_input[i * n_a + j];
            // std::cout << is_collide_b2w[i][j] << " ";
        }
        // std::cout << std::endl;
    }

    double L = (boundary_pos[2] - boundary_pos[0]) / 2.0;

    // Initialize reward_a matrix
    std::vector<double> reward_a(n_a, 0.0);

    // Inter-agent reward // penalize_distance
    if (condition[2]) {
        for (int agent = 0; agent < n_a; ++agent) {

            // std::vector<int> list_nei_1;
            // for (int i = 0; i < neighbor_index[agent].size(); ++i) {
            //     if (neighbor_index[agent][i] != -1) {
            //         list_nei_1.push_back(neighbor_index[agent][i]);
            //         // std::cout << neighbor_index[agent][i] << std::endl;
            //     }
            // }
            // std::vector<int> list_nei;
            // for (int i = 0; i < list_nei_1.size(); ++i) {
            //     if (list_nei_1[i] != leader_index_input[agent]) {
            //         list_nei.push_back(list_nei_1[i]);
            //         // std::cout << neighbor_index[agent][i] << std::endl;
            //     }
            // }

            std::vector<int> list_nei;
            for (int i = 0; i < neighbor_index[agent].size(); ++i) {
                if (neighbor_index[agent][i] != -1) {
                    list_nei.push_back(neighbor_index[agent][i]);
                    // std::cout << neighbor_index[agent][i] << std::endl;
                }
            }

            std::vector<double> pos_rel(2, 0.0);
            std::vector<double> avg_neigh_vel(2, 0.0);
            std::vector<double> avg_neigh_heading(2, 0.0);
            double dist_diff = 0.0;
            double dist_rel = 0.0;

            if (!list_nei.empty()) {
                for (int agent2 : list_nei) {
                    if (condition[0]) {
                        Matrix pos_rel_mat = {{matrix_p[0][agent2] - matrix_p[0][agent]}, {matrix_p[1][agent2] - matrix_p[1][agent]}};
                        _make_periodic(pos_rel_mat, L, boundary_pos, true);
                        pos_rel = _matrix_to_vector(pos_rel_mat);
                    } else {
                        pos_rel = {matrix_p[0][agent2] - matrix_p[0][agent], matrix_p[1][agent2] - matrix_p[1][agent]};
                    }
                    // std::cout << pos_rel[0] << ", " << pos_rel[1] << std::endl;

                    // dist_rel = _norm(pos_rel);
                    // if (dist_rel < d_ref) {
                    //     reward_a[agent] -= coefficients[0] * std::min(d_ref / (dist_rel + 1E-8) - 1.0, 20.0);
                    // } else {
                    //     reward_a[agent] -= coefficients[1] * (dist_rel - d_ref);
                    // }
                    reward_a[agent] -= std::max(coefficients[0] * (d_ref - _norm(pos_rel) - 0.05), coefficients[1] * (_norm(pos_rel) - d_ref - 0.05));
                    std::vector<double> dp_agent = _extract_column_one(matrix_dp, agent2);
                    avg_neigh_vel[0] += matrix_dp[0][agent2] / (_norm(dp_agent) + 1E-8);
                    avg_neigh_vel[1] += matrix_dp[1][agent2] / (_norm(dp_agent) + 1E-8);
                    
                }
                
                avg_neigh_vel[0] /= list_nei.size();
                avg_neigh_vel[1] /= list_nei.size();

                std::vector<double> dp_agent = _extract_column_one(matrix_dp, agent);
                double norm_dp_agent = _norm(dp_agent) + 1E-8;
                double vel_diff_norm = std::sqrt(std::pow(avg_neigh_vel[0] - matrix_dp[0][agent] / norm_dp_agent, 2) +
                                        std::pow(avg_neigh_vel[1] - matrix_dp[1][agent] / norm_dp_agent, 2));
                reward_a[agent] -= coefficients[2] * vel_diff_norm;

            }
        }
    }

    // Tracking virtual leader
    std::vector<double> leader_state(2*dim, 0.0);
    std::vector<double> leader_pos_rel(2, 0.0);
    std::vector<double> leader_vel_rel(2, 0.0);
    double const_dist_leader = 0.0;
    if (condition[3]) { // is_leader
        for (int agent = 0; agent < n_a; ++agent) {
            int nearest_leader_index = leader_index_input[agent];
            if (nearest_leader_index >= 0) {
                for (int i = 0; i < 2 * dim; ++i) {
                    leader_state[i] = matrix_leader_state[i][nearest_leader_index];
                }

                if (condition[0]) {
                    Matrix leader_agent_pos_rel = {{leader_state[0] - matrix_p[0][agent]}, {leader_state[1] - matrix_p[1][agent]}};
                    _make_periodic(leader_agent_pos_rel, L, boundary_pos, true);
                    leader_pos_rel = _matrix_to_vector(leader_agent_pos_rel);
                } else {
                    leader_pos_rel = {leader_state[0] - matrix_p[0][agent], leader_state[1] - matrix_p[1][agent]};
                }
                double norm_leader_p = std::sqrt(std::pow(leader_pos_rel[0], 2) + std::pow(leader_pos_rel[1], 2)); 
                if (norm_leader_p > const_dist_leader) {
                    reward_a[agent] -= coefficients[3] * (norm_leader_p - const_dist_leader);
                }

                leader_vel_rel = {leader_state[dim] - matrix_dp[0][agent], leader_state[dim + 1] - matrix_dp[1][agent]};
                double norm_leader_dp = std::sqrt(std::pow(leader_vel_rel[0], 2) + std::pow(leader_vel_rel[1], 2)); 
                reward_a[agent] -= coefficients[4] * norm_leader_dp;
                }
        }
    }

    // penalize_collide_agents
    if (condition[4]) {
        for (int agent = 0; agent < n_a; ++agent) {
            double sum = 0.0;
            for (int i = 0; i < n_a; ++i) {
                sum += is_collide_b2b[i][agent];
            }
            reward_a[agent] -= coefficients[5] * sum;
        }
    }

    // penalize_control_effort
    if (condition[5]) {
        for (int agent = 0; agent < n_a; ++agent) {
            double norm_a = 0.0;
            if (condition[1]) { // dynamics_mode == Cartesian
                norm_a = coefficients[6] * std::sqrt(std::pow(matrix_act[0][agent], 2) + std::pow(matrix_act[1][agent], 2));
            } else { // dynamics_mode == Polar
                norm_a = coefficients[7] * std::abs(matrix_act[0][agent]) + coefficients[8] * std::abs(matrix_act[1][agent]);
            }
            reward_a[agent] -= norm_a;
            // std::cout << coefficients[6] * norm_a << " ";
        }
        // std::cout << std::endl;
    }

    // penalize_collide_obstacles
    if (condition[6]) {
        for (int agent = 0; agent < n_a; ++agent) {
            double sum = 0.0;
            for (int i = n_a; i < n_a; ++i) {
                sum += is_collide_b2b[i][agent];
            }
            reward_a[agent] -= coefficients[9] * sum;
        }
    }

    // penalize_collide_walls
    if (condition[7]) {
        for (int agent = 0; agent < n_a; ++agent) {
            double sum = 0.0;
            for (int i = 0; i < 4; ++i) {
                sum += static_cast<double>(is_collide_b2w[i][agent]);
            }
            reward_a[agent] -= coefficients[10] * sum;
        }
    }

    for (int i = 0; i < n_a; ++i) {
        reward_input[i] = reward_a[i];
    }
}

std::tuple<Matrix, Matrix, std::vector<int>> _get_focused(Matrix Pos, 
                                                          Matrix Vel, 
                                                          double norm_threshold, 
                                                          int width, 
                                                          bool remove_self) 
{
    std::vector<double> norms(Pos[0].size());
    for (int i = 0; i < Pos[0].size(); ++i) {
        norms[i] = std::sqrt(Pos[0][i] * Pos[0][i] + Pos[1][i] * Pos[1][i]);
    }

    std::vector<int> sorted_seq(norms.size());
    std::iota(sorted_seq.begin(), sorted_seq.end(), 0);
    std::sort(sorted_seq.begin(), sorted_seq.end(), [&](int a, int b) { return norms[a] < norms[b]; });

    Matrix sorted_Pos(2, std::vector<double>(Pos[0].size()));
    for (int i = 0; i < Pos[0].size(); ++i) {
        sorted_Pos[0][i] = Pos[0][sorted_seq[i]];
        sorted_Pos[1][i] = Pos[1][sorted_seq[i]];
    }

    std::vector<double> sorted_norms(norms.size());
    for (int i = 0; i < norms.size(); ++i) {
        sorted_norms[i] = norms[sorted_seq[i]];
    }

    Matrix new_Pos;
    for (int i = 0; i < 2; ++i) {
        std::vector<double> col;
        for (int j = 0; j < sorted_Pos[0].size(); ++j) {
            if (sorted_norms[j] < norm_threshold) {
                col.push_back(sorted_Pos[i][j]);
            }
        }
        new_Pos.push_back(col);
    }

    std::vector<int> new_sorted_seq;
    for (int i = 0; i < sorted_Pos[0].size(); ++i) {
        if (sorted_norms[i] < norm_threshold) {
            new_sorted_seq.push_back(sorted_seq[i]);
        }
    }

    if (remove_self) {
        new_Pos[0].erase(new_Pos[0].begin());
        new_Pos[1].erase(new_Pos[1].begin());
        new_sorted_seq.erase(new_sorted_seq.begin());
    }

    Matrix new_Vel(2, std::vector<double>(new_sorted_seq.size()));
    for (int i = 0; i < new_sorted_seq.size(); ++i) {
        new_Vel[0][i] = Vel[0][new_sorted_seq[i]];
        new_Vel[1][i] = Vel[1][new_sorted_seq[i]];
    }

    Matrix target_Pos(2, std::vector<double>(width));
    Matrix target_Vel(2, std::vector<double>(width));

    size_t until_idx = std::min(new_Pos[0].size(), static_cast<size_t>(width));
    std::vector<int> target_Nei(until_idx, -1);
    for (int i = 0; i < until_idx; ++i) {
        target_Pos[0][i] = new_Pos[0][i];
        target_Pos[1][i] = new_Pos[1][i];
        target_Vel[0][i] = new_Vel[0][i];
        target_Vel[1][i] = new_Vel[1][i];
        target_Nei[i] = new_sorted_seq[i];
    }

    return std::make_tuple(target_Pos, target_Vel, target_Nei);
}

void _make_periodic(Matrix& x, double L, std::vector<double> bound_pos, bool is_rel) {
    
    if (is_rel) {
        for (int i = 0; i < x.size(); ++i) {
            for (int j = 0; j < x[i].size(); ++j) {
                // 如果元素大于 L，就减去 2*L
                if (x[i][j] > L)
                    x[i][j] -= 2 * L;
                // 如果元素小于 -L，就加上 2*L
                else if (x[i][j] < -L)
                    x[i][j] += 2 * L;
            }
        }
    } else {
        for (int j = 0; j < x[0].size(); ++j) {
            if (x[0][j] < bound_pos[0]) {
                x[0][j] += 2 * L;
            } else if (x[0][j] > bound_pos[2]) {
                x[0][j] -= 2 * L;
            }
            if (x[1][j] < bound_pos[3]) {
                x[1][j] += 2 * L;
            } else if (x[1][j] > bound_pos[1]) {
                x[1][j] -= 2 * L;
            }
        }
    }
    
}

void _sf_b2b_all(double *p_input,
                 double *sf_b2b_input, 
                 double *d_b2b_edge_input,
                 bool *is_collide_b2b_input,
                 double *boundary_pos_input,
                 double *d_b2b_center_input,
                 int n_a,
                 int dim,
                 double k_ball,
                 bool is_periodic)
{
    Matrix matrix_p(dim, std::vector<double>(n_a));
    Matrix matrix_d_b2b_edge(n_a, std::vector<double>(n_a));
    Matrix matrix_d_b2b_center(n_a, std::vector<double>(n_a));
    std::vector<std::vector<bool>> is_collide_b2b(n_a, std::vector<bool>(n_a, false));
    std::vector<double> boundary_pos(4, 0.0);

    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < n_a; ++j) {
            matrix_p[i][j] = p_input[i * n_a + j];
            // std::cout << matrix_p[i][j] << " ";
        }
        // std::cout << std::endl;
    }

    for (int i = 0; i < n_a; ++i) {
        for (int j = 0; j < n_a; ++j) {
            matrix_d_b2b_edge[i][j] = d_b2b_edge_input[i * n_a + j];
            matrix_d_b2b_center[i][j] = d_b2b_center_input[i * n_a + j];
            is_collide_b2b[i][j] = is_collide_b2b_input[i * n_a + j];
        }
    }

    for (int i = 0; i < 4; ++i) {
        boundary_pos[i] = boundary_pos_input[i];
    }

    Matrix sf_b2b_all(2 * n_a, std::vector<double>(n_a, 0.0));
    double L = (boundary_pos[2] - boundary_pos[0]) / 2.0;

    // 循环计算 sf_b2b_all
    for (int i = 0; i < n_a; ++i) {
        for (int j = 0; j < i; ++j) {
            Matrix delta = {
                {matrix_p[0][j] - matrix_p[0][i]},
                {matrix_p[1][j] - matrix_p[1][i]}
            };
            if (is_periodic) {
                _make_periodic(delta, L, boundary_pos, true);
            }

            double delta_x = delta[0][0] / matrix_d_b2b_center[i][j];
            double delta_y = delta[1][0] / matrix_d_b2b_center[i][j];
            sf_b2b_all[2 * i][j] = static_cast<double>(is_collide_b2b[i][j]) * matrix_d_b2b_edge[i][j] * k_ball * (-delta_x);
            sf_b2b_all[2 * i + 1][j] = static_cast<double>(is_collide_b2b[i][j]) * matrix_d_b2b_edge[i][j] * k_ball * (-delta_y);

            sf_b2b_all[2 * j][i] = -sf_b2b_all[2 * i][j];
            sf_b2b_all[2 * j + 1][i] = -sf_b2b_all[2 * i + 1][j];
            
            
        }
    }

    // 计算 sf_b2b
    Matrix sf_b2b(2, std::vector<double>(n_a));
    for (int i = 0; i < n_a; ++i) {
        for (int j = 0; j < dim; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n_a; ++k) {
                sum += sf_b2b_all[2 * i + j][k];
            }
            sf_b2b[j][i] = sum;
        }
    }

    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < n_a; ++j) {
           sf_b2b_input[i * n_a + j] = sf_b2b[i][j];
        }
    }

}

void _get_dist_b2w(double *p_input, 
                   double *r_input, 
                   double *d_b2w_input, 
                   bool *isCollision_input, 
                   int dim, 
                   int n_a, 
                   double *boundary_pos) 
{
    Matrix matrix_p(dim, std::vector<double>(n_a));
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < n_a; ++j) {
            matrix_p[i][j] = p_input[i * n_a + j];
        }
    }

    Matrix d_b2w(4, std::vector<double>(n_a, 0.0));
    std::vector<std::vector<bool>> isCollision(4, std::vector<bool>(n_a, false));
    
    for (int i = 0; i < n_a; ++i) {
        d_b2w[0][i] = matrix_p[0][i] - r_input[i] - boundary_pos[0];
        d_b2w[1][i] = boundary_pos[1] - (matrix_p[1][i] + r_input[i]);
        d_b2w[2][i] = boundary_pos[2] - (matrix_p[0][i] + r_input[i]);
        d_b2w[3][i] = matrix_p[1][i] - r_input[i] - boundary_pos[3];
    }
    
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < n_a; ++j) {
            isCollision[i][j] = (d_b2w[i][j] < 0);
            d_b2w[i][j] = std::abs(d_b2w[i][j]);
        }
    }

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < n_a; ++j) {
            d_b2w_input[i * n_a + j] = d_b2w[i][j];
            isCollision_input[i * n_a + j] = isCollision[i][j];
        }
    }
}

// 按行或按列拼接两个二维数组
Matrix _concatenate(const Matrix& arr1, const Matrix& arr2, int axis) {
    if (axis == 0) { // 按行拼接
        // 创建一个新的二维数组，行数为两个数组的行数之和，列数为第一个数组的列数
        Matrix result(arr1.size() + arr2.size(), std::vector<double>(arr1[0].size()));

        // 将arr1复制到结果数组中
        for (size_t i = 0; i < arr1.size(); ++i) {
            std::copy(arr1[i].begin(), arr1[i].end(), result[i].begin());
        }

        // 将arr2复制到结果数组中
        for (size_t i = 0; i < arr2.size(); ++i) {
            std::copy(arr2[i].begin(), arr2[i].end(), result[arr1.size() + i].begin());
        }

        return result;
    } else if (axis == 1) { // 按列拼接
        // 创建一个新的二维数组，行数为第一个数组的行数，列数为两个数组的列数之和
        Matrix result(arr1.size(), std::vector<double>(arr1[0].size() + arr2[0].size()));

        // 将arr1复制到结果数组中
        for (size_t i = 0; i < arr1.size(); ++i) {
            std::copy(arr1[i].begin(), arr1[i].end(), result[i].begin());
        }

        // 将arr2复制到结果数组中
        for (size_t i = 0; i < arr2.size(); ++i) {
            std::copy(arr2[i].begin(), arr2[i].end(), result[i].begin() + arr1[0].size());
        }

        return result;
    } else {
        // 如果axis参数不是0或1，则返回空数组
        return Matrix();
    }
}

// 提取二维数组的指定列，并返回一个二维数组
Matrix _extract_column(const Matrix& arr, size_t col_index) {
    Matrix result;

    // 检查索引是否有效
    if (col_index < arr[0].size()) {
        // 遍历二维数组的每一行，并提取指定列的数据作为一个新的行
        for (const auto& row : arr) {
            result.push_back({row[col_index]});
        }
    }
    return result;
}

std::vector<double> _extract_column_one(const Matrix& arr, size_t col_index) {
    std::vector<double> result;

    // 检查索引是否有效
    if (col_index < arr[0].size()) {
        // 遍历二维数组的每一行，并提取指定列的数据作为一个新的行
        for (const auto& row : arr) {
            result.push_back(row[col_index]);
        }
    }
    return result;
}

// Define a function to calculate _norm of a vector
double _norm(std::vector<double>& v) {
    double sum = 0.0;
    for (double x : v) {
        // sum += x * x;
        sum += std::pow(x, 2);
    }
    return std::sqrt(sum);
}

bool _all_elements_greater_than_(std::vector<int>& arr, int n_l) {
    for (int i = 0; i < arr.size(); ++i) {
        if (arr[i] <= (n_l - 1)) {
            return false;
        }
    }
    return true;
}

// Define a function to calculate cosine decay
double _rho_cos_dec(double z, double delta, double r) {
    if (z < delta * r) {
        return 1.0;
    } else if (z < r) {
        return (1.0 / 2.0) * (1.0 + std::cos(M_PI * (z / r - delta) / (1.0 - delta)));
    } else {
        return 0.0;
    }
}


Matrix _vector_to_matrix(const std::vector<double>& vec) {
    Matrix matrix(vec.size(), std::vector<double>(1));

    for (size_t i = 0; i < vec.size(); ++i) {
        matrix[i][0] = vec[i];
    }

    return matrix;
}

std::vector<double> _matrix_to_vector(const Matrix& matrix) {
    std::vector<double> vec;
    for (const auto& row : matrix) {
        for (const auto& element : row) {
            vec.push_back(element);
        }
    }
    return vec;
}
