#include "AdversarialEnv.h"
#include <iostream>
#include <tuple>
#include <algorithm>
#include <vector>
#include <cmath> 
#include <numeric>
#include <typeinfo>

// Custom clamp function
template <typename T>
T clamp(T value, T min, T max) {
    return std::max(min, std::min(value, max));
}

// Function to calculate observation for each environment
void _get_observation(double *p_input, 
                      double *dp_input, 
                      double *heading_input,
                      double *obs_input,
                      double *boundary_pos_input, 
                      double *hp_input,
                      int *index_l_input,
                      int *index_r_input,
                      double d_sen_l, 
                      double d_sen_r,
                      double hp_l_max,
                      double hp_r_max,
                      int topo_nei_l2l, 
                      int topo_nei_l2r,
                      int topo_nei_r2l, 
                      int topo_nei_r2r,
                      int n_l, 
                      int n_r,
                      int n_lr_init,
                      int obs_dim_agent, 
                      int dim, 
                      bool *condition) 
{
    int n_lr = n_l + n_r;
    
    // Initialize matrices for positions, velocities, headings, and observations
    Matrix matrix_p(dim, std::vector<double>(n_lr_init));
    Matrix matrix_dp(dim, std::vector<double>(n_lr_init));
    Matrix matrix_heading(dim, std::vector<double>(n_lr_init));
    Matrix matrix_obs(obs_dim_agent, std::vector<double>(n_lr));
    Matrix hp(1, std::vector<double>(n_lr_init));
    
    // Boundary positions and agent indices
    std::vector<double> boundary_pos(4, 0.0);
    std::vector<int> index_l(n_l, 0);
    std::vector<int> index_r(n_r, 0);

    // Copy position data from input arrays to matrices
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < n_lr_init; ++j) {
            matrix_p[i][j] = p_input[i * n_lr_init + j];
            matrix_dp[i][j] = dp_input[i * n_lr_init + j];
            matrix_heading[i][j] = heading_input[i * n_lr_init + j];
        }
    }
    
    // Copy health points data
    for (int i = 0; i < 1; ++i) {
        for (int j = 0; j < n_lr_init; ++j) {
            hp[i][j] = hp_input[i * n_lr_init + j];
        }
    }

    // Copy boundary positions
    for (int i = 0; i < 4; ++i) {
        boundary_pos[i] = boundary_pos_input[i];
    }

    // Copy agent indices for left and right groups
    for (int i = 0; i < n_l; ++i) {
        index_l[i] = index_l_input[i];
    }
    for (int i = 0; i < n_r; ++i) {
        index_r[i] = index_r_input[i];
    }

    // Calculate half-width of the boundary
    double L = (boundary_pos[2] - boundary_pos[0]) / 2.0;

    // Process left group agents
    for (int assemble_index_l = 0; assemble_index_l < n_l; ++assemble_index_l) {
        int agent_i = index_l[assemble_index_l];

        // Calculate relative positions and velocities from left agents to other left agents
        Matrix relPos_l2l(dim, std::vector<double>(n_l, 0.0));
        Matrix relVel_l2l(dim, std::vector<double>(n_l, 0.0));
        for (int j = 0; j < n_l; ++j) {
            for (int k = 0; k < dim; ++k) {
                relPos_l2l[k][j] = matrix_p[k][index_l[j]] - matrix_p[k][agent_i];
                if (condition[1]) {
                    // Use velocity difference for Cartesian dynamics
                    relVel_l2l[k][j] = matrix_dp[k][index_l[j]] - matrix_dp[k][agent_i];
                } else {
                    // Use heading difference for polar dynamics
                    relVel_l2l[k][j] = matrix_heading[k][index_l[j]] - matrix_heading[k][agent_i];
                }
            }
        }
        
        // Apply periodic boundary conditions if enabled
        if (condition[0]) {
            _make_periodic(relPos_l2l, L, boundary_pos, true);
        }
        
        // Get focused observations within sensing range
        std::tuple<Matrix, Matrix, std::vector<int>> focused_obs_l2l = _get_focused(relPos_l2l, relVel_l2l, d_sen_l, topo_nei_l2l, true);
        Matrix relPos_l2l_focused = std::get<0>(focused_obs_l2l);
        Matrix relVel_l2l_focused = std::get<1>(focused_obs_l2l);

        // Calculate relative positions and velocities from left agents to right agents
        Matrix relPos_l2r(dim, std::vector<double>(n_r, 0.0));
        Matrix relVel_l2r(dim, std::vector<double>(n_r, 0.0));
        for (int j = 0; j < n_r; ++j) {
            for (int k = 0; k < dim; ++k) {
                relPos_l2r[k][j] = matrix_p[k][index_r[j]] - matrix_p[k][agent_i];
                if (condition[1]) {
                    relVel_l2r[k][j] = matrix_dp[k][index_r[j]] - matrix_dp[k][agent_i];
                } else {
                    relVel_l2r[k][j] = matrix_heading[k][index_r[j]] - matrix_heading[k][agent_i];
                }
            }
        }
        
        // Apply periodic boundary conditions if enabled
        if (condition[0]) {
            _make_periodic(relPos_l2r, L, boundary_pos, true);
        }
        
        // Get focused observations within sensing range
        std::tuple<Matrix, Matrix, std::vector<int>> focused_obs_l2r = _get_focused(relPos_l2r, relVel_l2r, d_sen_l, topo_nei_l2r, false);
        Matrix relPos_l2r_focused = std::get<0>(focused_obs_l2r);
        Matrix relVel_l2r_focused = std::get<1>(focused_obs_l2r);

        // Concatenate observations
        Matrix obs_agent;
        if (condition[2]) { // Include agent's own state in observation
            Matrix obs_l_pos = _concatenate(_extract_column(matrix_p, agent_i), relPos_l2l_focused, 1);
            Matrix obs_l_vel = _concatenate(_extract_column(matrix_dp, agent_i), relVel_l2l_focused, 1);
            Matrix obs_l_pos_2 = _concatenate(obs_l_pos, relPos_l2r_focused, 1);
            Matrix obs_l_vel_2 = _concatenate(obs_l_vel, relVel_l2r_focused, 1);
            obs_agent = _concatenate(obs_l_pos_2, obs_l_vel_2, 0);
        } else {
            Matrix obs_l_pos_2 = _concatenate(relPos_l2l_focused, relPos_l2r_focused, 1);
            Matrix obs_l_vel_2 = _concatenate(relVel_l2l_focused, relVel_l2r_focused, 1);
            obs_agent = _concatenate(obs_l_pos_2, obs_l_vel_2, 0);
        }

        // Transpose obs_agent matrix and flatten to 1D array
        std::vector<double> obs_agent_flat;
        obs_agent_flat.reserve(obs_agent.size() * obs_agent[0].size());
        for (size_t j = 0; j < obs_agent[0].size(); ++j) {
            for (size_t i = 0; i < obs_agent.size(); ++i) {
                obs_agent_flat.push_back(obs_agent[i][j]);
            }
        }

        if (condition[1]) { // Cartesian dynamics mode
            for (int j = 0; j < obs_dim_agent; ++j) {
                matrix_obs[j][assemble_index_l] = obs_agent_flat[j];
            }
        } else { // Polar dynamics mode
            for (int j = 0; j < obs_dim_agent - dim; ++j) {
                matrix_obs[j][assemble_index_l] = obs_agent_flat[j];
            }
            // Add heading information for polar dynamics
            for (int j = 0; j < dim; ++j) {
                matrix_obs[obs_dim_agent - dim + j][assemble_index_l] = matrix_heading[j][agent_i];
            }
        }
    }

    // Process right group agents
    for (int assemble_index_r = n_l; assemble_index_r < n_lr; ++assemble_index_r) {
        int agent_i = index_r[assemble_index_r - n_l];

        // Calculate relative positions and velocities from right agents to left agents
        Matrix relPos_r2l(dim, std::vector<double>(n_l, 0.0));
        Matrix relVel_r2l(dim, std::vector<double>(n_l, 0.0));
        for (int j = 0; j < n_l; ++j) {
            for (int k = 0; k < dim; ++k) {
                relPos_r2l[k][j] = matrix_p[k][index_l[j]] - matrix_p[k][agent_i];
                if (condition[1]) {
                    relVel_r2l[k][j] = matrix_dp[k][index_l[j]] - matrix_dp[k][agent_i];
                } else {
                    relVel_r2l[k][j] = matrix_heading[k][index_l[j]] - matrix_heading[k][agent_i];
                }
            }
        }
        
        // Apply periodic boundary conditions if enabled
        if (condition[0]) {
            _make_periodic(relPos_r2l, L, boundary_pos, true);
        }
        
        // Get focused observations within sensing range
        std::tuple<Matrix, Matrix, std::vector<int>> focused_obs_r2l = _get_focused(relPos_r2l, relVel_r2l, d_sen_r, topo_nei_r2l, false);
        Matrix relPos_r2l_focused = std::get<0>(focused_obs_r2l);
        Matrix relVel_r2l_focused = std::get<1>(focused_obs_r2l);

        // Calculate relative positions and velocities from right agents to other right agents
        Matrix relPos_r2r(dim, std::vector<double>(n_r, 0.0));
        Matrix relVel_r2r(dim, std::vector<double>(n_r, 0.0));
        for (int j = 0; j < n_r; ++j) {
            for (int k = 0; k < dim; ++k) {
                relPos_r2r[k][j] = matrix_p[k][index_r[j]] - matrix_p[k][agent_i];
                if (condition[1]) {
                    relVel_r2r[k][j] = matrix_dp[k][index_r[j]] - matrix_dp[k][agent_i];
                } else {
                    relVel_r2r[k][j] = matrix_heading[k][index_r[j]] - matrix_heading[k][agent_i];
                }
            }
        }
        
        // Apply periodic boundary conditions if enabled
        if (condition[0]) {
            _make_periodic(relPos_r2r, L, boundary_pos, true);
        }
        
        // Get focused observations within sensing range
        std::tuple<Matrix, Matrix, std::vector<int>> focused_obs_r2r = _get_focused(relPos_r2r, relVel_r2r, d_sen_r, topo_nei_r2r, true);
        Matrix relPos_r2r_focused = std::get<0>(focused_obs_r2r);
        Matrix relVel_r2r_focused = std::get<1>(focused_obs_r2r);

        // Concatenate observations
        Matrix obs_agent;
        if (condition[2]) { // Include agent's own state in observation
            Matrix obs_escaper_pos = _concatenate(_extract_column(matrix_p, agent_i), relPos_r2r_focused, 1);
            Matrix obs_escaper_vel = _concatenate(_extract_column(matrix_dp, agent_i), relVel_r2r_focused, 1);
            Matrix obs_escaper_pos_2 = _concatenate(obs_escaper_pos, relPos_r2l_focused, 1);
            Matrix obs_escaper_vel_2 = _concatenate(obs_escaper_vel, relVel_r2l_focused, 1);
            obs_agent = _concatenate(obs_escaper_pos_2, obs_escaper_vel_2, 0);
        } else {
            Matrix obs_escaper_pos_2 = _concatenate(relPos_r2r_focused, relPos_r2l_focused, 1);
            Matrix obs_escaper_vel_2 = _concatenate(relVel_r2r_focused, relVel_r2l_focused, 1);
            obs_agent = _concatenate(obs_escaper_pos_2, obs_escaper_vel_2, 0);
        }

        // Transpose obs_agent matrix and flatten to 1D array
        std::vector<double> obs_agent_flat;
        obs_agent_flat.reserve(obs_agent.size() * obs_agent[0].size());
        for (size_t j = 0; j < obs_agent[0].size(); ++j) {
            for (size_t i = 0; i < obs_agent.size(); ++i) {
                obs_agent_flat.push_back(obs_agent[i][j]);
            }
        }

        if (condition[1]) { // Cartesian dynamics mode
            for (int j = 0; j < obs_dim_agent; ++j) {
                matrix_obs[j][assemble_index_r] = obs_agent_flat[j];
            }
        } else { // Polar dynamics mode
            for (int j = 0; j < obs_dim_agent - dim; ++j) {
                matrix_obs[j][assemble_index_r] = obs_agent_flat[j];
            }
            // Add heading information for polar dynamics
            for (int j = 0; j < dim; ++j) {
                matrix_obs[obs_dim_agent - dim + j][assemble_index_r] = matrix_heading[j][agent_i];
            }
        }
    }

    // Copy final observation matrix to output array
    for (int i = 0; i < obs_dim_agent; ++i) {
        for (int j = 0; j < n_lr; ++j) {
            obs_input[i * n_lr + j] = matrix_obs[i][j];
        }
    }
}

// Function to calculate rewards for left and right agents based on various conditions
void _get_reward(double *act_input, 
                 double *boundary_pos_input, 
                 double *hp_input,
                 double *reward_l_input,
                 double *reward_r_input, 
                 double *coefficients,
                 bool *condition, 
                 bool *is_collide_b2b_input, 
                 bool *is_collide_b2w_input,
                 int *index_l_input,
                 int *index_r_input,
                 int *attack_nei_index_input,
                 int *safe_nei_index_input,
                 int attack_max, 
                 int safe_max,
                 int n_l, 
                 int n_r,
                 int n_lr_init,
                 int dim) 
{ 
    // Total number of agents (left + right)
    int n_lr = n_l + n_r;
    
    // Initialize action matrix and neighbor indices
    Matrix matrix_act(dim, std::vector<double>(n_lr));
    std::vector<std::vector<int>> attack_neigh(n_lr_init, std::vector<int>(attack_max, -1));
    std::vector<std::vector<int>> safe_neigh(n_lr_init, std::vector<int>(safe_max, -1));
    std::vector<double> boundary_pos(4, 0.0);
    
    // Collision matrices: bot-to-bot and bot-to-wall
    std::vector<std::vector<bool>> is_collide_b2b(n_lr_init, std::vector<bool>(n_lr_init, false));
    std::vector<std::vector<bool>> is_collide_b2w(4, std::vector<bool>(n_lr_init, false));
    
    // Health points matrix
    Matrix hp(1, std::vector<double>(n_lr_init));
    std::vector<int> index_l(n_l, 0);
    std::vector<int> index_r(n_r, 0);

    // Parse input action matrix
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < n_lr; ++j) {
            matrix_act[i][j] = act_input[i * n_lr + j];
        }
    }

    // Parse attack neighbor indices
    for (int i = 0; i < n_lr_init; ++i) {
        for (int j = 0; j < attack_max; ++j) {
            attack_neigh[i][j] = attack_nei_index_input[i * attack_max + j];
        }
    }
    
    // Parse safe neighbor indices
    for (int i = 0; i < n_lr_init; ++i) {
        for (int j = 0; j < safe_max; ++j) {
            safe_neigh[i][j] = safe_nei_index_input[i * safe_max + j];
        }
    }

    // Parse boundary positions
    for (int i = 0; i < 4; ++i) {
        boundary_pos[i] = boundary_pos_input[i];
    }

    // Parse bot-to-bot collision matrix
    for (int i = 0; i < n_lr_init; ++i) {
        for (int j = 0; j < n_lr_init; ++j) {
            is_collide_b2b[i][j] = is_collide_b2b_input[i * n_lr_init + j];
        }
    }
    
    // Parse bot-to-wall collision matrix
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < n_lr_init; ++j) {
            is_collide_b2w[i][j] = is_collide_b2w_input[i * n_lr_init + j];
        }
    }

    // Parse health points
    for (int i = 0; i < 1; ++i) {
        for (int j = 0; j < n_lr_init; ++j) {
            hp[i][j] = hp_input[i * n_lr_init + j];
        }
    }

    // Parse left and right agent indices
    for (int i = 0; i < n_l; ++i) {
        index_l[i] = index_l_input[i];
    }
    for (int i = 0; i < n_r; ++i) {
        index_r[i] = index_r_input[i];
    }
    
    // Concatenate left and right indices
    std::vector<int> index_concat;
    index_concat.reserve(index_l.size() + index_r.size()); // Reserve sufficient space
    index_concat.insert(index_concat.end(), index_l.begin(), index_l.end());
    index_concat.insert(index_concat.end(), index_r.begin(), index_r.end());

    // Calculate half boundary length
    double L = (boundary_pos[2] - boundary_pos[0]) / 2.0;
    
    // Initialize reward vectors
    std::vector<double> reward_l(n_l, 0.0);
    std::vector<double> reward_r(n_r, 0.0);

    // Lambda function to train agents and calculate situational rewards
    auto train_agent = [&](int num_agents, const std::vector<int>& indices, 
                           std::vector<double>& reward_self, std::vector<double>& reward_other) {
        for (int i = 0; i < num_agents; ++i) {
            int agent_index = indices[i];
            
            // Situation reward - process attack neighbors
            std::vector<int> list_nei;
            for (int nei : attack_neigh[agent_index]) {
                if (nei != -1) {
                    list_nei.push_back(nei);
                }
            }
            // Reward other agents for being attack neighbors
            for (int nei : list_nei) {
                reward_other[nei] += coefficients[1];
            }

            // Process safe neighbors
            std::vector<int> list_safe_nei;
            for (int nei : safe_neigh[agent_index]) {
                if (nei != -1) {
                    list_safe_nei.push_back(nei);
                }
            }
            // Penalize other agents for being safe neighbors
            for (int nei : list_safe_nei) {
                reward_other[nei] -= coefficients[2];
            }

            // Kill reward - penalize dead agents and reward their attackers
            if (hp[0][agent_index] <= 0) {
                reward_self[i] -= coefficients[0];
                for (int nei : list_nei) {
                    reward_other[nei] += coefficients[0];
                }
            }
        }
    };

    // Apply training rewards based on conditions
    if (condition[6]) { // Train right agent
        train_agent(n_l, index_l, reward_l, reward_r);
    }
    if (condition[5]) { // Train left agent
        train_agent(n_r, index_r, reward_r, reward_l);
    }
    
    // Penalize control effort
    if (condition[0]) {
        // Penalize left agents for control effort
        for (int agent = 0; agent < n_l; ++agent) {
            double norm_a = 0.0;
            if (condition[4]) { // Cartesian dynamics mode
                norm_a = coefficients[3] * std::sqrt(std::pow(matrix_act[0][agent], 2) + std::pow(matrix_act[1][agent], 2));
            } else { // Polar dynamics mode
                norm_a = coefficients[4] * std::abs(matrix_act[0][agent]) + coefficients[5] * std::abs(matrix_act[1][agent]);
            }
            reward_l[agent] -= norm_a;
        }
        
        // Penalize right agents for control effort
        for (int agent = n_l; agent < n_lr; ++agent) {
            double norm_b = 0.0;
            if (condition[4]) { // Cartesian dynamics mode
                norm_b = coefficients[3] * std::sqrt(std::pow(matrix_act[0][agent], 2) + std::pow(matrix_act[1][agent], 2));
            } else { // Polar dynamics mode
                norm_b = coefficients[4] * std::abs(matrix_act[0][agent]) + coefficients[5] * std::abs(matrix_act[1][agent]);
            }
            reward_r[agent - n_l] -= norm_b;
        }
    }
    
    // Penalize collisions between agents on the same side
    if (condition[1]) {
        // Create active collision matrix for current agents
        std::vector<std::vector<bool>> is_collide_b2b_active(index_concat.size(), std::vector<bool>(index_concat.size()));
        for (size_t i = 0; i < index_concat.size(); ++i) {
            for (size_t j = 0; j < index_concat.size(); ++j) {
                is_collide_b2b_active[i][j] = is_collide_b2b[index_concat[i]][index_concat[j]];
            }
        }
        
        // Penalize left agents for colliding with other left agents
        for (int agent = 0; agent < n_l; ++agent) {
            double sum_a = 0.0;
            for (int i = 0; i < n_l; ++i) {
                sum_a += static_cast<double>(is_collide_b2b_active[i][agent]);
            }
            reward_l[agent] -= coefficients[6] * sum_a;
        }
        
        // Penalize right agents for colliding with other right agents
        for (int agent = n_l; agent < n_lr; ++agent) {
            double sum_b = 0.0;
            for (int i = n_l; i < n_lr; ++i) {
                sum_b += static_cast<double>(is_collide_b2b_active[i][agent]);
            }
            reward_r[agent - n_l] -= coefficients[7] * sum_b;
        }
    }

    // Penalize collisions with opponents
    if (condition[2]) {
        // Create active collision matrix for opponent collisions
        std::vector<std::vector<bool>> is_collide_b2b_active_op(index_concat.size(), std::vector<bool>(index_concat.size()));
        for (size_t i = 0; i < index_concat.size(); ++i) {
            for (size_t j = 0; j < index_concat.size(); ++j) {
                is_collide_b2b_active_op[i][j] = is_collide_b2b[index_concat[i]][index_concat[j]];
            }
        }
        
        // Penalize left agents for colliding with right agents
        for (int agent = 0; agent < n_l; ++agent) {
            double sum_a_op = 0.0;
            for (int i = n_l; i < n_lr; ++i) {
                sum_a_op += static_cast<double>(is_collide_b2b_active_op[i][agent]);
            }
            reward_l[agent] -= coefficients[8] * sum_a_op;
        }
        
        // Penalize right agents for colliding with left agents
        for (int agent = n_l; agent < n_lr; ++agent) {
            double sum_b_op = 0.0;
            for (int i = 0; i < n_l; ++i) {
                sum_b_op += static_cast<double>(is_collide_b2b_active_op[i][agent]);
            }
            reward_r[agent - n_l] -= coefficients[9] * sum_b_op;
        }
    }

    // Penalize collisions with walls
    if (condition[3]) {
        // Create active wall collision matrix
        std::vector<std::vector<bool>> is_collide_b2w_active(index_concat.size(), std::vector<bool>(index_concat.size()));
        for (size_t i = 0; i < 4; ++i) {
            for (size_t j = 0; j < index_concat.size(); ++j) {
                is_collide_b2w_active[i][j] = is_collide_b2w[i][index_concat[j]];
            }
        }
        
        // Penalize left agents for wall collisions
        for (int agent = 0; agent < n_l; ++agent) {
            double sum_a = 0.0;
            for (int i = 0; i < 4; ++i) {
                sum_a += static_cast<double>(is_collide_b2w_active[i][agent]);
            }
            reward_l[agent] -= coefficients[10] * sum_a;
        }
        
        // Penalize right agents for wall collisions
        for (int agent = n_l; agent < n_lr; ++agent) {
            double sum_b = 0.0;
            for (int i = 0; i < 4; ++i) {
                sum_b += static_cast<double>(is_collide_b2w_active[i][agent]);
            }
            reward_r[agent - n_l] -= coefficients[11] * sum_b;
        }
    }

    // Copy calculated rewards to output arrays
    for (int i = 0; i < n_l; ++i) {
        reward_l_input[i] = reward_l[i];
    }
    for (int i = 0; i < n_r; ++i) {
        reward_r_input[i] = reward_r[i];
    }
}

// Function to get focused agents based on distance threshold and maximum width
std::tuple<Matrix, Matrix, std::vector<int>> _get_focused(Matrix Pos, 
                                                          Matrix Vel, 
                                                          double norm_threshold, 
                                                          int width, 
                                                          bool remove_self) 
{
    // Calculate norms (distances) for all positions
    std::vector<double> norms(Pos[0].size());
    for (int i = 0; i < Pos[0].size(); ++i) {
        norms[i] = std::sqrt(Pos[0][i] * Pos[0][i] + Pos[1][i] * Pos[1][i]);
    }

    // Create sorted sequence based on distance
    std::vector<int> sorted_seq(norms.size());
    std::iota(sorted_seq.begin(), sorted_seq.end(), 0);
    std::sort(sorted_seq.begin(), sorted_seq.end(), [&](int a, int b) { return norms[a] < norms[b]; });

    // Sort positions by distance
    Matrix sorted_Pos(2, std::vector<double>(Pos[0].size()));
    for (int i = 0; i < Pos[0].size(); ++i) {
        sorted_Pos[0][i] = Pos[0][sorted_seq[i]];
        sorted_Pos[1][i] = Pos[1][sorted_seq[i]];
    }

    // Sort norms accordingly
    std::vector<double> sorted_norms(norms.size());
    for (int i = 0; i < norms.size(); ++i) {
        sorted_norms[i] = norms[sorted_seq[i]];
    }

    // Filter positions within threshold
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

    // Filter sorted sequence within threshold
    std::vector<int> new_sorted_seq;
    for (int i = 0; i < sorted_Pos[0].size(); ++i) {
        if (sorted_norms[i] < norm_threshold) {
            new_sorted_seq.push_back(sorted_seq[i]);
        }
    }

    // Remove self if requested
    if (remove_self) {
        new_Pos[0].erase(new_Pos[0].begin());
        new_Pos[1].erase(new_Pos[1].begin());
        new_sorted_seq.erase(new_sorted_seq.begin());
    }

    // Create corresponding velocity matrix
    Matrix new_Vel(2, std::vector<double>(new_sorted_seq.size()));
    for (int i = 0; i < new_sorted_seq.size(); ++i) {
        new_Vel[0][i] = Vel[0][new_sorted_seq[i]];
        new_Vel[1][i] = Vel[1][new_sorted_seq[i]];
    }

    // Create target matrices with specified width
    Matrix target_Pos(2, std::vector<double>(width));
    Matrix target_Vel(2, std::vector<double>(width));

    // Fill target matrices up to width limit
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

// Process action data by mapping indices and extracting relevant values
void _process_act(int *index_l_last_input,
                  int *index_r_last_input,
                  int *index_l_input,
                  int *index_r_input,
                  double *a_com_input,
                  double *a_true_input,
                  double *a_input,
                  int dim,
                  int n_lr,
                  int n_lr_init,
                  int n_l_last,
                  int n_r_last,
                  int n_l,
                  int n_r) 
{
    int n_lr_last = n_l_last + n_r_last;
    
    // Initialize matrices for combined, true, and current actions
    Matrix a_com(dim, std::vector<double>(n_lr_init));
    Matrix a_true(dim, std::vector<double>(n_lr));
    Matrix a(dim, std::vector<double>(n_lr_last));
    
    // Initialize index vectors for left and right sides
    std::vector<int> index_l_last(n_l_last, 0);
    std::vector<int> index_r_last(n_r_last, 0);
    std::vector<int> index_l(n_l, 0);
    std::vector<int> index_r(n_r, 0);

    // Convert input array to matrix format
    for (int j = 0; j < n_lr_last; ++j) {
        for (int i = 0; i < dim; ++i) {
            a[i][j] = a_input[j * dim + i];
        }
    }

    // Copy index arrays from input
    for (int i = 0; i < n_l_last; ++i) {
        index_l_last[i] = index_l_last_input[i];
    }
    for (int i = 0; i < n_r_last; ++i) {
        index_r_last[i] = index_r_last_input[i];
    }
    for (int i = 0; i < n_l; ++i) {
        index_l[i] = index_l_input[i];
    }
    for (int i = 0; i < n_r; ++i) {
        index_r[i] = index_r_input[i];
    }
    
    // Process left side indices - find matching indices and copy data
    for (int a_i = 0; a_i < n_l; ++a_i) {
        auto it = std::find(index_l_last.begin(), index_l_last.end(), index_l[a_i]);
        if (it != index_l_last.end()) {
            int act_index = std::distance(index_l_last.begin(), it);
            for (size_t row = 0; row < a.size(); ++row) {
                a_com[row][index_l[a_i]] = a[row][act_index];
                a_true[row][a_i] = a[row][act_index];
            }
        }
    }

    // Process right side indices - find matching indices and copy data
    for (int a_i = 0; a_i < n_r; ++a_i) {
        auto it = std::find(index_r_last.begin(), index_r_last.end(), index_r[a_i]);
        if (it != index_r_last.end()) {
            int act_index = std::distance(index_r_last.begin(), it) + index_l_last.size();
            for (size_t row = 0; row < a.size(); ++row) {
                a_com[row][index_r[a_i]] = a[row][act_index];
                a_true[row][a_i + n_l] = a[row][act_index];
            }
        }
    }

    // Convert matrices back to output arrays
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < n_lr_init; ++j) {
            a_com_input[i * n_lr_init + j] = a_com[i][j];
        }
    }
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < n_lr; ++j) {
            a_true_input[i * n_lr + j] = a_true[i][j];
        }
    }
}

// Process attack mechanics including neighbor detection and health updates
void _process_attack(int *index_l_input,
                     int *index_r_input,
                     int *dead_index_input,
                     int *attack_neigh_input,
                     int *safe_neigh_input,
                     double *p_input,
                     double *dp_input,
                     double *hp_input,
                     double *boundary_pos_input,
                     bool *is_training,
                     bool is_periodic,
                     double attack_radius,
                     double attack_angle,
                     double attack_hp,
                     double recover_hp,
                     int attack_max,
                     int safe_max,
                     int n_lr_init,
                     int n_l,
                     int n_r,
                     int dim) 
{
    // Initialize matrices for positions, velocities, and health points
    Matrix matrix_p(dim, std::vector<double>(n_lr_init));
    Matrix matrix_dp(dim, std::vector<double>(n_lr_init));
    Matrix hp(1, std::vector<double>(n_lr_init));
    
    // Initialize index vectors and neighbor lists
    std::vector<int> index_l(n_l, 0);
    std::vector<int> index_r(n_r, 0);
    std::vector<std::vector<int>> attack_neigh(n_lr_init, std::vector<int>(attack_max, -1));
    std::vector<std::vector<int>> safe_neigh(n_lr_init, std::vector<int>(safe_max, -1));
    std::vector<double> boundary_pos(4, 0.0);

    // Convert input arrays to matrix format
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < n_lr_init; ++j) {
            matrix_p[i][j] = p_input[i * n_lr_init + j];
            matrix_dp[i][j] = dp_input[i * n_lr_init + j];
        }
    }
    for (int i = 0; i < 1; ++i) {
        for (int j = 0; j < n_lr_init; ++j) {
            hp[i][j] = hp_input[i * n_lr_init + j];
        }
    }
    
    // Copy index arrays
    for (int i = 0; i < n_l; ++i) {
        index_l[i] = index_l_input[i];
    }
    for (int i = 0; i < n_r; ++i) {
        index_r[i] = index_r_input[i];
    }

    // Copy boundary positions and calculate half-length
    for (int i = 0; i < 4; ++i) {
        boundary_pos[i] = boundary_pos_input[i];
    }
    double L = (boundary_pos[2] - boundary_pos[0]) / 2.0;

    // Lambda function to process attack mechanics for one side
    auto process_side = [&](const std::vector<int>& victims, const std::vector<int>& opponents) {
        for (int a_i : victims) {
            // Calculate relative positions between victim and all opponents
            Matrix pos_rel(dim, std::vector<double>(opponents.size()));
            for (size_t j = 0; j < opponents.size(); ++j) {
                for (size_t k = 0; k < dim; ++k) {
                    pos_rel[k][j] = matrix_p[k][opponents[j]] - matrix_p[k][a_i];
                }
            }

            // Apply periodic boundary conditions if enabled
            if (is_periodic) {
                _make_periodic(pos_rel, L, boundary_pos, true);
            }

            // Calculate distances to all opponents
            std::vector<double> pos_rel_norm(opponents.size());
            for (size_t j = 0; j < opponents.size(); ++j) {
                std::vector<double> col(dim);
                for (size_t k = 0; k < dim; ++k) {
                    col[k] = pos_rel[k][j];
                }
                pos_rel_norm[j] = _norm(col);
            }

            // Sort opponents by distance
            std::vector<size_t> sorted_seq(pos_rel_norm.size());
            std::iota(sorted_seq.begin(), sorted_seq.end(), 0);
            std::sort(sorted_seq.begin(), sorted_seq.end(), [&](size_t a, size_t b) {
                return pos_rel_norm[a] < pos_rel_norm[b];
            });

            // Find opponents within attack radius
            std::vector<size_t> threat_index_pos;
            for (size_t j : sorted_seq) {
                if (pos_rel_norm[j] < attack_radius) {
                    threat_index_pos.push_back(j);
                }
            }

            // Initialize directional threat and safe indices
            std::vector<size_t> threat_index_dir_p;
            std::vector<size_t> safe_index_dir_p;
            std::vector<size_t> threat_index_dir_v;
            std::vector<size_t> safe_index_dir_v;
            std::vector<double> dp_col;
            std::vector<double> p_op_col;
            std::vector<double> v_op_col;
            double dot_products = 0.0;
            double norm_v = 0.0;
            double norm_op = 0.0;
            double norm_ov = 0.0;
            double cos_angle = 0.0;
            double angle = 0.0;
            dp_col = _extract_column_one(matrix_dp, a_i);
            norm_v = _norm(dp_col);

            // Calculate angle between agent's velocity and opponents' relative position
            for (size_t k = 0; k < opponents.size(); ++k) {
                p_op_col = _extract_column_one(pos_rel, k);

                dot_products = _dot_product(dp_col, p_op_col);
                norm_op = _norm(p_op_col);
                cos_angle = dot_products / (norm_v * norm_op + 1E-8);
                angle = std::acos(clamp(cos_angle, -1.0, 1.0));

                if (angle > ((1.0 - attack_angle) * M_PI)) {
                    threat_index_dir_p.push_back(k);
                }
                if (angle < (attack_angle * M_PI)) {
                    safe_index_dir_p.push_back(k);
                }
            }

            // Calculate angle between agent's velocity and opponents' relative velocity
            for (size_t k = 0; k < opponents.size(); ++k) {
                v_op_col = _extract_column_one(matrix_dp, opponents[k]);
                p_op_col = _extract_column_one(pos_rel, k);

                dot_products = _dot_product(p_op_col, v_op_col);
                norm_ov = _norm(v_op_col);
                norm_op = _norm(p_op_col);
                cos_angle = -dot_products / (norm_op * norm_ov + 1E-8);
                angle = std::acos(clamp(cos_angle, -1.0, 1.0));

                if (angle < (attack_angle * M_PI)) {
                    threat_index_dir_v.push_back(k);
                }
                if (angle > ((1.0 - attack_angle) * M_PI)) {
                    safe_index_dir_v.push_back(k);
                }
            }
            
            // Find final threat and safe indices by intersecting position and direction criteria
            std::vector<size_t> threat_index_dir = _intersect(threat_index_dir_p, threat_index_dir_v);
            std::vector<size_t> safe_index_dir = _intersect(safe_index_dir_p, safe_index_dir_v);
            std::vector<size_t> threat_index = _intersect_and_sort(threat_index_pos, threat_index_dir);
            std::vector<size_t> safe_index = _intersect_and_sort(threat_index_pos, safe_index_dir);

            // Process attack if threats exist, otherwise recover health
            if (!threat_index.empty()) {
                int num_attack = std::min(static_cast<int>(threat_index.size()), attack_max);
                for (int j = 0; j < num_attack; ++j) {
                    attack_neigh[a_i][j] = threat_index[j];
                }
                hp[0][a_i] -= attack_hp * static_cast<double>(num_attack);
                // hp[0][opponents[threat_index[0]]] -= attack_hp;
            } else {
                hp[0][a_i] += recover_hp;
            }
            
            // Record safe neighbors
            if (!safe_index.empty()) {
                int num_safe = static_cast<int>(safe_index.size());
                for (int j = 0; j < num_safe; ++j) {
                    safe_neigh[a_i][j] = safe_index[j];
                }
            }

            // Mark agent as dead if health drops to zero or below
            if (hp[0][a_i] <= 0) {
                dead_index_input[a_i] = a_i;
            }
        }
    };

    // Process attacks for both sides
    // if (is_training[1]) {
    process_side(index_l, index_r);
    // }
    // if (is_training[0]) {
    process_side(index_r, index_l);
    // }

    // Convert matrices back to output arrays
    for (int i = 0; i < 1; ++i) {
        for (int j = 0; j < n_lr_init; ++j) {
            hp_input[i * n_lr_init + j] = hp[i][j];
        }
    }
    for (int i = 0; i < n_lr_init; ++i) {
        for (int j = 0; j < attack_max; ++j) {
            attack_neigh_input[i * attack_max + j] = attack_neigh[i][j];
        }
    }
    for (int i = 0; i < n_lr_init; ++i) {
        for (int j = 0; j < safe_max; ++j) {
            safe_neigh_input[i * safe_max + j] = safe_neigh[i][j];
        }
    }
}

// Apply periodic boundary conditions to matrix elements
void _make_periodic(Matrix& x, double L, std::vector<double> bound_pos, bool is_rel) {
    
    if (is_rel) {
        // Handle relative positions
        for (int i = 0; i < x.size(); ++i) {
            for (int j = 0; j < x[i].size(); ++j) {
                // If element is greater than L, subtract 2*L
                if (x[i][j] > L)
                    x[i][j] -= 2 * L;
                // If element is less than -L, add 2*L
                else if (x[i][j] < -L)
                    x[i][j] += 2 * L;
            }
        }
    } else {
        // Handle absolute positions with boundary wrapping
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

// Ball-to-ball interaction force calculation
void _sf_b2b_all(double *p_input,
                 double *sf_b2b_input, 
                 double *d_b2b_edge_input,
                 bool *is_collide_b2b_input,
                 double *boundary_pos_input,
                 double *d_b2b_center_input,
                 int *dead_index_input,
                 int n_lr_init,
                 int dim,
                 double k_ball,
                 bool is_periodic)
{
    // Initialize position matrix from flattened array
    Matrix matrix_p(dim, std::vector<double>(n_lr_init));
    // Initialize distance matrices for edge-to-edge and center-to-center
    Matrix matrix_d_b2b_edge(n_lr_init, std::vector<double>(n_lr_init));
    Matrix matrix_d_b2b_center(n_lr_init, std::vector<double>(n_lr_init));
    // Collision status matrix initialization
    std::vector<std::vector<bool>> is_collide_b2b(n_lr_init, std::vector<bool>(n_lr_init, false));
    // Boundary positions (x_min, y_min, x_max, y_max)
    std::vector<double> boundary_pos(4, 0.0);
    // Dead agent indices
    std::vector<int> dead_index(n_lr_init, -1);

    // Fill position matrix from 1D input array
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < n_lr_init; ++j) {
            matrix_p[i][j] = p_input[i * n_lr_init + j];
        }
    }

    // Fill distance matrices and collision status from 1D inputs
    for (int i = 0; i < n_lr_init; ++i) {
        for (int j = 0; j < n_lr_init; ++j) {
            matrix_d_b2b_edge[i][j] = d_b2b_edge_input[i * n_lr_init + j];
            matrix_d_b2b_center[i][j] = d_b2b_center_input[i * n_lr_init + j];
            is_collide_b2b[i][j] = is_collide_b2b_input[i * n_lr_init + j];
        }
    }

    // Set boundary positions
    for (int i = 0; i < 4; ++i) {
        boundary_pos[i] = boundary_pos_input[i];
    }

    // Set dead indices
    for (int i = 0; i < n_lr_init; ++i) {
        dead_index[i] = dead_index_input[i];
    }

    // Initialize force matrix for all pairs
    Matrix sf_b2b_all(2 * n_lr_init, std::vector<double>(n_lr_init, 0.0));
    // Calculate periodic domain half-length
    double L = (boundary_pos[2] - boundary_pos[0]) / 2.0;

    // Calculate pair-wise forces
    for (int i = 0; i < n_lr_init; ++i) {
        for (int j = 0; j < i; ++j) {
            // Position difference vector
            Matrix delta = {
                {matrix_p[0][j] - matrix_p[0][i]},
                {matrix_p[1][j] - matrix_p[1][i]}
            };
            // Apply periodic boundary adjustment if needed
            if (is_periodic) {
                _make_periodic(delta, L, boundary_pos, true);
            }

            // Skip force calculation for dead agents
            auto it = std::find(dead_index.begin(), dead_index.end(), i);
            if (it != dead_index.end()){
                sf_b2b_all[2 * i][j] = 0.0;
                sf_b2b_all[2 * i + 1][j] = 0.0;

                sf_b2b_all[2 * j][i] = 0.0;
                sf_b2b_all[2 * j + 1][i] = 0.0;
            } else {
                // Normalized direction components
                double delta_x = delta[0][0] / (matrix_d_b2b_center[i][j] + 1E-8);
                double delta_y = delta[1][0] / (matrix_d_b2b_center[i][j] + 1E-8);
                // Calculate repulsive force components
                sf_b2b_all[2 * i][j] = static_cast<double>(is_collide_b2b[i][j]) * matrix_d_b2b_edge[i][j] * k_ball * (-delta_x);
                sf_b2b_all[2 * i + 1][j] = static_cast<double>(is_collide_b2b[i][j]) * matrix_d_b2b_edge[i][j] * k_ball * (-delta_y);
                // Apply Newton's third law (equal and opposite force)
                sf_b2b_all[2 * j][i] = -sf_b2b_all[2 * i][j];
                sf_b2b_all[2 * j + 1][i] = -sf_b2b_all[2 * i + 1][j];
            }
        }
    }

    // Calculate net force per agent
    Matrix sf_b2b(2, std::vector<double>(n_lr_init));
    for (int i = 0; i < n_lr_init; ++i) {
        for (int j = 0; j < dim; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n_lr_init; ++k) {
                sum += sf_b2b_all[2 * i + j][k];
            }
            sf_b2b[j][i] = sum;
        }
    }

    // Flatten force matrix to 1D output array
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < n_lr_init; ++j) {
           sf_b2b_input[i * n_lr_init + j] = sf_b2b[i][j];
        }
    }
}

// Calculate ball-to-wall distances and collision status
void _get_dist_b2w(double *p_input, 
                   double *r_input, 
                   double *d_b2w_input, 
                   bool *isCollision_input, 
                   int dim, 
                   int n_pe, 
                   double *boundary_pos) 
{
    // Initialize position matrix
    Matrix matrix_p(dim, std::vector<double>(n_pe));
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < n_pe; ++j) {
            matrix_p[i][j] = p_input[i * n_pe + j];
        }
    }

    // Initialize distance matrix (4 walls x n_pe agents)
    Matrix d_b2w(4, std::vector<double>(n_pe, 0.0));
    // Collision status matrix
    std::vector<std::vector<bool>> isCollision(4, std::vector<bool>(n_pe, false));
    
    // Calculate distances to walls:
    // [0] = left wall, [1] = top wall
    // [2] = right wall, [3] = bottom wall
    for (int i = 0; i < n_pe; ++i) {
        d_b2w[0][i] = matrix_p[0][i] - r_input[i] - boundary_pos[0];   // Left wall
        d_b2w[1][i] = boundary_pos[1] - (matrix_p[1][i] + r_input[i]); // Top wall
        d_b2w[2][i] = boundary_pos[2] - (matrix_p[0][i] + r_input[i]); // Right wall
        d_b2w[3][i] = matrix_p[1][i] - r_input[i] - boundary_pos[3];   // Bottom wall
    }
    
    // Determine collision status and absolute distances
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < n_pe; ++j) {
            isCollision[i][j] = (d_b2w[i][j] < 0);
            d_b2w[i][j] = std::abs(d_b2w[i][j]);
        }
    }

    // Flatten results to 1D arrays
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < n_pe; ++j) {
            d_b2w_input[i * n_pe + j] = d_b2w[i][j];
            isCollision_input[i * n_pe + j] = isCollision[i][j];
        }
    }
}

// Concatenate two matrices along specified axis
Matrix _concatenate(const Matrix& arr1, const Matrix& arr2, int axis) {
    if (axis == 0) { // Row-wise concatenation
        Matrix result(arr1.size() + arr2.size(), std::vector<double>(arr1[0].size()));
        for (size_t i = 0; i < arr1.size(); ++i) {
            std::copy(arr1[i].begin(), arr1[i].end(), result[i].begin());
        }
        for (size_t i = 0; i < arr2.size(); ++i) {
            std::copy(arr2[i].begin(), arr2[i].end(), result[arr1.size() + i].begin());
        }
        return result;
    } else if (axis == 1) { // Column-wise concatenation
        Matrix result(arr1.size(), std::vector<double>(arr1[0].size() + arr2[0].size()));
        for (size_t i = 0; i < arr1.size(); ++i) {
            std::copy(arr1[i].begin(), arr1[i].end(), result[i].begin());
        }
        for (size_t i = 0; i < arr2.size(); ++i) {
            std::copy(arr2[i].begin(), arr2[i].end(), result[i].begin() + arr1[0].size());
        }
        return result;
    } else {
        return Matrix();  // Invalid axis
    }
}

// Extract specified column as a new matrix
Matrix _extract_column(const Matrix& arr, size_t col_index) {
    Matrix result;
    if (col_index < arr[0].size()) {
        for (const auto& row : arr) {
            result.push_back({row[col_index]});
        }
    }
    return result;
}

// Extract specified column as 1D vector
std::vector<double> _extract_column_one(const Matrix& arr, size_t col_index) {
    std::vector<double> result;
    if (col_index < arr[0].size()) {
        for (const auto& row : arr) {
            result.push_back(row[col_index]);
        }
    }
    return result;
}

// Compute dot product of two vectors
double _dot_product(const std::vector<double>& a, const std::vector<double>& b) {
    double result = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}

// Compute Euclidean norm of vector
double _norm(std::vector<double>& v) {
    double sum = 0.0;
    for (double x : v) {
        sum += std::pow(x, 2);
    }
    return std::sqrt(sum);
}

// Cosine decay function for smooth transitions
double _rho_cos_dec(double z, double delta, double r) {
    if (z < delta * r) return 1.0;
    else if (z < r) return 0.5 * (1.0 + std::cos(M_PI * (z/r - delta)/(1.0 - delta)));
    else return 0.0;
}

// Convert 1D vector to column matrix
Matrix _vector_to_matrix(const std::vector<double>& vec) {
    Matrix matrix(vec.size(), std::vector<double>(1));
    for (size_t i = 0; i < vec.size(); ++i) {
        matrix[i][0] = vec[i];
    }
    return matrix;
}

// Convert matrix to 1D vector (column-major order)
std::vector<double> _matrix_to_vector(const Matrix& matrix) {
    std::vector<double> vec;
    for (const auto& row : matrix) {
        for (const auto& element : row) {
            vec.push_back(element);
        }
    }
    return vec;
}

// Find sorted intersection of two index vectors
std::vector<size_t> _intersect_and_sort(const std::vector<size_t>& v1, const std::vector<size_t>& v2) {
    std::vector<size_t> common_elements;
    std::vector<size_t> pos_indices;

    for (size_t i = 0; i < v1.size(); ++i) {
        if (std::find(v2.begin(), v2.end(), v1[i]) != v2.end()) {
            common_elements.push_back(v1[i]);
            pos_indices.push_back(i);
        }
    }

    // Maintain original ordering from first vector
    std::vector<size_t> threat_index(common_elements.size());
    std::iota(threat_index.begin(), threat_index.end(), 0);
    std::sort(threat_index.begin(), threat_index.end(), [&](size_t i, size_t j) {
        return pos_indices[i] < pos_indices[j];
    });

    for (size_t i = 0; i < threat_index.size(); ++i) {
        threat_index[i] = common_elements[threat_index[i]];
    }

    return threat_index;
}

// Find unordered intersection of two vectors
std::vector<size_t> _intersect(const std::vector<size_t>& v1, const std::vector<size_t>& v2) {
    std::vector<size_t> threat_index;
    for (size_t i = 0; i < v1.size(); ++i) {
        if (std::find(v2.begin(), v2.end(), v1[i]) != v2.end()) {
            threat_index.push_back(v1[i]);
        }
    }
    return threat_index;
}
