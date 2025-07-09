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
    // Initialize matrices for states and observations
    Matrix matrix_p(dim, std::vector<double>(n_a));
    Matrix matrix_dp(dim, std::vector<double>(n_a));
    Matrix matrix_heading(dim, std::vector<double>(n_a));
    Matrix matrix_leader_state(2 * dim, std::vector<double>(n_l));
    Matrix matrix_obs(obs_dim_agent, std::vector<double>(n_a));
    std::vector<std::vector<int>> neighbor_index(n_a, std::vector<int>(topo_nei_max, -1));
    std::vector<double> boundary_pos(4, 0.0);

    // Populate position/velocity/heading matrices from input arrays
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < n_a; ++j) {
            matrix_p[i][j] = p_input[i * n_a + j];
            matrix_dp[i][j] = dp_input[i * n_a + j];
            matrix_heading[i][j] = heading_input[i * n_a + j];
        }
    }

    // Populate leader state matrix
    for (int i = 0; i < 2 * dim; ++i) {
        for (int j = 0; j < n_l; ++j) {
            matrix_leader_state[i][j] = leader_state_input[i * n_l + j];
        }
    }

    // Set boundary positions
    for (int i = 0; i < 4; ++i) {
        boundary_pos[i] = boundary_pos_input[i];
    }

    double L = (boundary_pos[2] - boundary_pos[0]) / 2.0;  // Calculate domain half-length
    std::vector<double> shill_agent_state(4, 0.0);  // Placeholder for special agent state
    
    // Process each agent individually
    for (int agent_i = 0; agent_i < n_a; ++agent_i) {
        // Initialize relative position/velocity matrices
        Matrix relPos_a2a(dim, std::vector<double>(n_a, 0.0));
        Matrix relVel_a2a(dim, std::vector<double>(n_a, 0.0));

        // Calculate relative states to current agent
        for (int j = 0; j < n_a; ++j) {
            for (int k = 0; k < dim; ++k) {
                relPos_a2a[k][j] = matrix_p[k][j] - matrix_p[k][agent_i];
                // Determine velocity based on dynamics mode
                if (condition[3]) {
                    relVel_a2a[k][j] = matrix_dp[k][j] - matrix_dp[k][agent_i];
                } else {
                    relVel_a2a[k][j] = matrix_heading[k][j] - matrix_heading[k][agent_i];
                }
            }
        }

        // Apply periodic boundary conditions if enabled
        if (condition[0]) {
            _make_periodic(relPos_a2a, L, boundary_pos, true);
        }

        // Focus on neighbors within sensing range
        std::tuple<Matrix, Matrix, std::vector<int>> focused_obs = _get_focused(relPos_a2a, relVel_a2a, d_sen, topo_nei_max, true);
        Matrix relPos_a2a_focused = std::get<0>(focused_obs);
        Matrix relVel_a2a_focused = std::get<1>(focused_obs);
        std::vector<int> nei_index = std::get<2>(focused_obs);
        
        // Record neighbor indices
        for (int i = 0; i < nei_index.size(); ++i) {
            neighbor_index[agent_i][i] = nei_index[i];
        }

        // Apply neighbor permutation if enabled
        Matrix relPos_a2a_permutated = relPos_a2a_focused;
        Matrix relVel_a2a_permutated = relVel_a2a_focused;
        if (condition[4]) { 
            for (size_t i = 0; i < relPos_a2a_focused.size(); ++i) {
                for (size_t j = 0; j < topo_nei_max; ++j) {
                    relPos_a2a_permutated[i][j] = relPos_a2a_focused[i][random_permutation[j]];
                    relVel_a2a_permutated[i][j] = relVel_a2a_focused[i][random_permutation[j]];
                }
            }
        }

        // Compose agent observation
        Matrix obs_agent;
        if (condition[5]) { // Include self-state
            Matrix obs_agent_pos = _concatenate(_extract_column(matrix_p, agent_i), relPos_a2a_permutated, 1);
            Matrix obs_agent_vel = _concatenate(_extract_column(matrix_dp, agent_i), relVel_a2a_permutated, 1);
            obs_agent = _concatenate(obs_agent_pos, obs_agent_vel, 0);
        } else { // Exclude self-state
            obs_agent = _concatenate(relPos_a2a_permutated, relVel_a2a_permutated, 0);
        }

        // Flatten observation matrix to 1D array
        std::vector<double> obs_agent_flat;
        obs_agent_flat.reserve(obs_agent.size() * obs_agent[0].size());
        for (size_t j = 0; j < obs_agent[0].size(); ++j) {
            for (size_t i = 0; i < obs_agent.size(); ++i) {
                obs_agent_flat.push_back(obs_agent[i][j]);
            }
        }

        // Process leader information if conditions met
        if (condition[1] && condition[2]) { 
            // Initialize leader-relative states
            std::vector<double> leader_pos_rel_flat(dim, 0.0);
            std::vector<double> leader_vel_rel_flat(dim, 0.0);
            std::vector<double> leader_heading_rel_flat(dim, 0.0);

            // Calculate leader relative positions
            Matrix leader_pos_rel_mat(dim, std::vector<double>(n_l));
            for (int j = 0; j < n_l; ++j) {
                for (int k = 0; k < dim; ++k) {
                    leader_pos_rel_mat[k][j] = matrix_leader_state[k][j] - matrix_p[k][agent_i];
                }
            }

            // Apply periodic boundaries to leader positions
            if (condition[0]) {
                _make_periodic(leader_pos_rel_mat, L, boundary_pos, true);
            }

            // Find nearest leader
            int min_index = 0;
            if (n_l > 1) {
                std::vector<double> leader_agent_norms;
                for (size_t j = 0; j < leader_pos_rel_mat[0].size(); ++j) {
                    double norm_value = 0.0;
                    for (size_t i = 0; i < leader_pos_rel_mat.size(); ++i) {
                        norm_value += std::pow(leader_pos_rel_mat[i][j], 2);
                    }
                    norm_value = std::sqrt(norm_value);
                    leader_agent_norms.push_back(norm_value);
                }
                auto min_it = std::min_element(leader_agent_norms.begin(), leader_agent_norms.end());
                min_index = std::distance(leader_agent_norms.begin(), min_it);
            }

            std::vector<double> leader_vel(dim, 0.0);
            double norm_value_ = 0.0;
            for (size_t i = 0; i < leader_pos_rel_mat.size(); ++i) {
                norm_value_ += std::pow(leader_pos_rel_mat[i][min_index], 2);
            }
            norm_value_ = std::sqrt(norm_value_);
            
            // Only include leader within sensing range
            if (norm_value_ < d_sen) {
                leader_index_input[agent_i] = min_index;  // Record nearest leader index
                for (int j = 0; j < dim; ++j) {
                    leader_pos_rel_flat[j] = leader_pos_rel_mat[j][min_index];
                    leader_vel_rel_flat[j] = matrix_leader_state[dim + j][min_index] - matrix_dp[j][agent_i];
                }
                // Calculate relative heading
                for (size_t j = 0; j < dim; ++j) {
                    leader_vel[j] = matrix_leader_state[dim + j][min_index];
                }
                double leader_vel_norm = _norm(leader_vel) + 1E-8;  // Avoid division by zero
                for (int j = 0; j < dim; ++j) {
                    leader_heading_rel_flat[j] = (matrix_leader_state[dim + j][min_index] / leader_vel_norm - matrix_heading[j][agent_i]);
                }
            }

            // Assemble observations based on dynamics mode
            if (condition[3]) { // Cartesian dynamics
                for (int j = 0; j < obs_dim_agent - 2 * dim; ++j) {
                    matrix_obs[j][agent_i] = obs_agent_flat[j];
                }
                // Append leader position and velocity
                for (int j = 0; j < dim; ++j) {
                    matrix_obs[obs_dim_agent - 2 * dim + j][agent_i] = leader_pos_rel_flat[j];
                }
                for (int j = 0; j < dim; ++j) {
                    matrix_obs[obs_dim_agent - dim + j][agent_i] = leader_vel_rel_flat[j];
                }
            } else { // Polar dynamics
                for (int j = 0; j < obs_dim_agent - 3 * dim; ++j) {
                    matrix_obs[j][agent_i] = obs_agent_flat[j];
                }
                // Append leader position, heading, and self-heading
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
        } else {  // Non-leader agents
            if (condition[3]) { // Cartesian dynamics
                for (int j = 0; j < obs_dim_agent; ++j) {
                    matrix_obs[j][agent_i] = obs_agent_flat[j];
                }
            } else { // Polar dynamics
                for (int j = 0; j < obs_dim_agent - dim; ++j) {
                    matrix_obs[j][agent_i] = obs_agent_flat[j];
                }
                // Append self-heading
                for (int j = 0; j < dim; ++j) {
                    matrix_obs[obs_dim_agent - dim + j][agent_i] = matrix_heading[j][agent_i];
                }
            }
        }
    }

    // Convert observation matrix to 1D output array
    for (int i = 0; i < obs_dim_agent; ++i) {
        for (int j = 0; j < n_a; ++j) {
            obs_input[i * n_a + j] = matrix_obs[i][j];
        }
    }

    // Convert neighbor indices to 1D output array
    for (int i = 0; i < n_a; ++i) {
        for (int j = 0; j < topo_nei_max; ++j) {
            neighbor_index_input[i * topo_nei_max + j] = neighbor_index[i][j];
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
    // Initialize matrices from input arrays
    Matrix matrix_p(dim, std::vector<double>(n_a));
    Matrix matrix_dp(dim, std::vector<double>(n_a));
    Matrix matrix_heading(dim, std::vector<double>(n_a));
    Matrix matrix_act(dim, std::vector<double>(n_a));
    Matrix matrix_leader_state(2 * dim, std::vector<double>(n_l));
    std::vector<std::vector<int>> neighbor_index(n_a, std::vector<int>(topo_nei_max, -1));
    std::vector<double> boundary_pos(4, 0.0);
    std::vector<std::vector<bool>> is_collide_b2b(n_a, std::vector<bool>(n_a, false));
    std::vector<std::vector<bool>> is_collide_b2w(4, std::vector<bool>(n_a, false));

    // Populate position, velocity, and heading matrices
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < n_a; ++j) {
            matrix_p[i][j] = p_input[i * n_a + j];
            matrix_dp[i][j] = dp_input[i * n_a + j];
            matrix_heading[i][j] = heading_input[i * n_a + j];
        }
    }

    // Populate action matrix (note dimension ordering)
    for (int j = 0; j < n_a; ++j) {
        for (int i = 0; i < dim; ++i) {
            matrix_act[i][j] = act_input[j * dim + i];
        }
    }

    // Populate leader state matrix
    for (int i = 0; i < 2 * dim; ++i) {
        for (int j = 0; j < n_l; ++j) {
            matrix_leader_state[i][j] = leader_state_input[i * n_l + j];
        }
    }

    // Populate neighbor indices from input
    for (int i = 0; i < n_a; ++i) {
        for (int j = 0; j < topo_nei_max; ++j) {
            neighbor_index[i][j] = neighbor_index_input[i * topo_nei_max + j];
        }
    }

    // Set boundary positions
    for (int i = 0; i < 4; ++i) {
        boundary_pos[i] = boundary_pos_input[i];
    }

    // Populate collision status matrices
    for (int i = 0; i < n_a; ++i) {
        for (int j = 0; j < n_a; ++j) {
            is_collide_b2b[i][j] = is_collide_b2b_input[i * n_a + j];
        }
    }
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < n_a; ++j) {
            is_collide_b2w[i][j] = is_collide_b2w_input[i * n_a + j];
        }
    }

    double L = (boundary_pos[2] - boundary_pos[0]) / 2.0;  // Calculate domain half-length
    
    // Initialize per-agent rewards
    std::vector<double> reward_a(n_a, 0.0);

    // ========== INTER-AGENT REWARDS SECTION ==========
    if (condition[2]) {  // Check if inter-agent rewards are enabled
        for (int agent = 0; agent < n_a; ++agent) {
            // Identify neighbors within sensing range
            std::vector<int> list_nei;
            for (int i = 0; i < neighbor_index[agent].size(); ++i) {
                if (neighbor_index[agent][i] != -1) {
                    list_nei.push_back(neighbor_index[agent][i]);
                }
            }

            std::vector<double> pos_rel(2, 0.0);
            std::vector<double> avg_neigh_vel(2, 0.0);
            double dist_diff = 0.0;
            double dist_rel = 0.0;

            if (!list_nei.empty()) {
                // Process each neighbor
                for (int agent2 : list_nei) {
                    // Calculate relative position (with periodic adjustment if enabled)
                    if (condition[0]) {
                        Matrix pos_rel_mat = {{matrix_p[0][agent2] - matrix_p[0][agent]}, 
                                             {matrix_p[1][agent2] - matrix_p[1][agent]}};
                        _make_periodic(pos_rel_mat, L, boundary_pos, true);
                        pos_rel = _matrix_to_vector(pos_rel_mat);
                    } else {
                        pos_rel = {matrix_p[0][agent2] - matrix_p[0][agent], 
                                  matrix_p[1][agent2] - matrix_p[1][agent]};
                    }

                    // Penalize distance deviation from reference
                    double distance = _norm(pos_rel);
                    double distance_penalty = (d_ref - distance > 0) ? 
                        coefficients[0] * (d_ref - distance - 0.05) : 
                        coefficients[1] * (distance - d_ref - 0.05);
                    reward_a[agent] -= distance_penalty;

                    // Aggregate neighbor velocity information
                    std::vector<double> dp_agent = _extract_column_one(matrix_dp, agent2);
                    avg_neigh_vel[0] += matrix_dp[0][agent2] / (_norm(dp_agent) + 1E-8);
                    avg_neigh_vel[1] += matrix_dp[1][agent2] / (_norm(dp_agent) + 1E-8);
                }
                
                // Calculate average neighbor velocity
                avg_neigh_vel[0] /= list_nei.size();
                avg_neigh_vel[1] /= list_nei.size();

                // Calculate velocity alignment penalty
                std::vector<double> dp_agent = _extract_column_one(matrix_dp, agent);
                double norm_dp_agent = _norm(dp_agent) + 1E-8;
                double vel_diff_norm = std::sqrt(
                    std::pow(avg_neigh_vel[0] - matrix_dp[0][agent] / norm_dp_agent, 2) +
                    std::pow(avg_neigh_vel[1] - matrix_dp[1][agent] / norm_dp_agent, 2)
                );
                reward_a[agent] -= coefficients[2] * vel_diff_norm;
            }
        }
    }

    // ========== LEADER TRACKING REWARDS SECTION ==========
    std::vector<double> leader_state(2*dim, 0.0);
    std::vector<double> leader_pos_rel(2, 0.0);
    std::vector<double> leader_vel_rel(2, 0.0);
    double const_dist_leader = 0.0;  // Constant distance threshold
    
    if (condition[3]) {  // Leader tracking enabled
        for (int agent = 0; agent < n_a; ++agent) {
            int nearest_leader_index = leader_index_input[agent];
            if (nearest_leader_index >= 0) {
                // Get leader state
                for (int i = 0; i < 2 * dim; ++i) {
                    leader_state[i] = matrix_leader_state[i][nearest_leader_index];
                }

                // Calculate leader-relative position
                if (condition[0]) {
                    Matrix leader_agent_pos_rel = {
                        {leader_state[0] - matrix_p[0][agent]}, 
                        {leader_state[1] - matrix_p[1][agent]}
                    };
                    _make_periodic(leader_agent_pos_rel, L, boundary_pos, true);
                    leader_pos_rel = _matrix_to_vector(leader_agent_pos_rel);
                } else {
                    leader_pos_rel = {
                        leader_state[0] - matrix_p[0][agent], 
                        leader_state[1] - matrix_p[1][agent]
                    };
                }
                
                // Calculate position tracking penalty
                double norm_leader_p = std::sqrt(
                    std::pow(leader_pos_rel[0], 2) + 
                    std::pow(leader_pos_rel[1], 2)
                ); 
                if (norm_leader_p > const_dist_leader) {
                    reward_a[agent] -= coefficients[3] * (norm_leader_p - const_dist_leader);
                }

                // Calculate velocity tracking penalty
                leader_vel_rel = {
                    leader_state[dim] - matrix_dp[0][agent],
                    leader_state[dim + 1] - matrix_dp[1][agent]
                };
                double norm_leader_dp = std::sqrt(
                    std::pow(leader_vel_rel[0], 2) + 
                    std::pow(leader_vel_rel[1], 2)
                ); 
                reward_a[agent] -= coefficients[4] * norm_leader_dp;
            }
        }
    }

    // ========== COLLISION PENALTY SECTION ==========
    if (condition[4]) {  // Agent-agent collision penalties
        for (int agent = 0; agent < n_a; ++agent) {
            double sum = 0.0;
            // Count collisions with other agents
            for (int i = 0; i < n_a; ++i) {
                sum += is_collide_b2b[i][agent];
            }
            reward_a[agent] -= coefficients[5] * sum;
        }
    }

    // ========== CONTROL EFFORT PENALTY SECTION ==========
    if (condition[5]) {  // Control effort penalty
        for (int agent = 0; agent < n_a; ++agent) {
            double norm_a = 0.0;
            if (condition[1]) {  // Cartesian dynamics
                norm_a = coefficients[6] * std::sqrt(
                    std::pow(matrix_act[0][agent], 2) + 
                    std::pow(matrix_act[1][agent], 2)
                );
            } else {  // Polar dynamics
                norm_a = coefficients[7] * std::abs(matrix_act[0][agent]) + 
                         coefficients[8] * std::abs(matrix_act[1][agent]);
            }
            reward_a[agent] -= norm_a;
        }
    }

    // ========== OBSTACLE COLLISION PENALTY SECTION ==========
    if (condition[6]) {  // Obstacle collision penalty (currently unimplemented)
        for (int agent = 0; agent < n_a; ++agent) {
            double sum = 0.0;
            for (int i = n_a; i < n_a; ++i) {  // Note: range appears empty
                sum += is_collide_b2b[i][agent];
            }
            reward_a[agent] -= coefficients[9] * sum;
        }
    }

    // ========== WALL COLLISION PENALTY SECTION ==========
    if (condition[7]) {  // Wall collision penalty
        for (int agent = 0; agent < n_a; ++agent) {
            double sum = 0.0;
            // Count collisions with walls (4 boundaries)
            for (int i = 0; i < 4; ++i) {
                sum += static_cast<double>(is_collide_b2w[i][agent]);
            }
            reward_a[agent] -= coefficients[10] * sum;
        }
    }

    // Output results to reward_input array
    for (int i = 0; i < n_a; ++i) {
        reward_input[i] = reward_a[i];
    }
}

// Filters positions/velocities within norm_threshold and returns top 'width' elements
std::tuple<Matrix, Matrix, std::vector<int>> _get_focused(Matrix Pos, 
                                                          Matrix Vel, 
                                                          double norm_threshold, 
                                                          int width, 
                                                          bool remove_self) 
{
    // Calculate Euclidean norms for each position vector
    std::vector<double> norms(Pos[0].size());
    for (int i = 0; i < Pos[0].size(); ++i) {
        norms[i] = std::sqrt(Pos[0][i] * Pos[0][i] + Pos[1][i] * Pos[1][i]);
    }

    // Sort indices by norm value (ascending)
    std::vector<int> sorted_seq(norms.size());
    std::iota(sorted_seq.begin(), sorted_seq.end(), 0);
    std::sort(sorted_seq.begin(), sorted_seq.end(), [&](int a, int b) { return norms[a] < norms[b]; });

    // Create sorted position matrix
    Matrix sorted_Pos(2, std::vector<double>(Pos[0].size()));
    for (int i = 0; i < Pos[0].size(); ++i) {
        sorted_Pos[0][i] = Pos[0][sorted_seq[i]];
        sorted_Pos[1][i] = Pos[1][sorted_seq[i]];
    }

    // Create sorted norm vector
    std::vector<double> sorted_norms(norms.size());
    for (int i = 0; i < norms.size(); ++i) {
        sorted_norms[i] = norms[sorted_seq[i]];
    }

    // Filter positions below threshold
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

    // Keep indices of filtered positions
    std::vector<int> new_sorted_seq;
    for (int i = 0; i < sorted_Pos[0].size(); ++i) {
        if (sorted_norms[i] < norm_threshold) {
            new_sorted_seq.push_back(sorted_seq[i]);
        }
    }

    // Remove self-reference if requested
    if (remove_self) {
        new_Pos[0].erase(new_Pos[0].begin());
        new_Pos[1].erase(new_Pos[1].begin());
        new_sorted_seq.erase(new_sorted_seq.begin());
    }

    // Create velocity matrix for filtered neighbors
    Matrix new_Vel(2, std::vector<double>(new_sorted_seq.size()));
    for (int i = 0; i < new_sorted_seq.size(); ++i) {
        new_Vel[0][i] = Vel[0][new_sorted_seq[i]];
        new_Vel[1][i] = Vel[1][new_sorted_seq[i]];
    }

    // Prepare output matrices with fixed width
    Matrix target_Pos(2, std::vector<double>(width));
    Matrix target_Vel(2, std::vector<double>(width));

    // Fill output matrices with top neighbors
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

// Applies periodic boundary conditions to position vectors
void _make_periodic(Matrix& x, double L, std::vector<double> bound_pos, bool is_rel) {
    
    if (is_rel) {  // Handle relative positions
        for (int i = 0; i < x.size(); ++i) {
            for (int j = 0; j < x[i].size(); ++j) {
                // Wrap values outside [-L, L] range
                if (x[i][j] > L)
                    x[i][j] -= 2 * L;
                else if (x[i][j] < -L)
                    x[i][j] += 2 * L;
            }
        }
    } else {  // Handle absolute positions
        for (int j = 0; j < x[0].size(); ++j) {
            // Wrap x-coordinate
            if (x[0][j] < bound_pos[0]) {
                x[0][j] += 2 * L;
            } else if (x[0][j] > bound_pos[2]) {
                x[0][j] -= 2 * L;
            }
            // Wrap y-coordinate
            if (x[1][j] < bound_pos[3]) {
                x[1][j] += 2 * L;
            } else if (x[1][j] > bound_pos[1]) {
                x[1][j] -= 2 * L;
            }
        }
    }
    
}

// Computes spring forces between colliding agents
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
    // Initialize matrices from input arrays
    Matrix matrix_p(dim, std::vector<double>(n_a));
    Matrix matrix_d_b2b_edge(n_a, std::vector<double>(n_a));
    Matrix matrix_d_b2b_center(n_a, std::vector<double>(n_a));
    std::vector<std::vector<bool>> is_collide_b2b(n_a, std::vector<bool>(n_a, false));
    std::vector<double> boundary_pos(4, 0.0);

    // Populate position matrix
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < n_a; ++j) {
            matrix_p[i][j] = p_input[i * n_a + j];
        }
    }

    // Populate collision-related matrices
    for (int i = 0; i < n_a; ++i) {
        for (int j = 0; j < n_a; ++j) {
            matrix_d_b2b_edge[i][j] = d_b2b_edge_input[i * n_a + j];
            matrix_d_b2b_center[i][j] = d_b2b_center_input[i * n_a + j];
            is_collide_b2b[i][j] = is_collide_b2b_input[i * n_a + j];
        }
    }

    // Set boundary positions
    for (int i = 0; i < 4; ++i) {
        boundary_pos[i] = boundary_pos_input[i];
    }

    // Initialize force matrix
    Matrix sf_b2b_all(2 * n_a, std::vector<double>(n_a, 0.0));
    double L = (boundary_pos[2] - boundary_pos[0]) / 2.0;  // Domain half-length

    // Compute pairwise collision forces
    for (int i = 0; i < n_a; ++i) {
        for (int j = 0; j < i; ++j) {
            // Calculate relative position
            Matrix delta = {
                {matrix_p[0][j] - matrix_p[0][i]},
                {matrix_p[1][j] - matrix_p[1][i]}
            };
            // Apply periodic boundaries if enabled
            if (is_periodic) {
                _make_periodic(delta, L, boundary_pos, true);
            }

            // Calculate force components
            double delta_x = delta[0][0] / matrix_d_b2b_center[i][j];
            double delta_y = delta[1][0] / matrix_d_b2b_center[i][j];
            // Calculate spring forces (action-reaction pair)
            sf_b2b_all[2 * i][j] = static_cast<double>(is_collide_b2b[i][j]) * matrix_d_b2b_edge[i][j] * k_ball * (-delta_x);
            sf_b2b_all[2 * i + 1][j] = static_cast<double>(is_collide_b2b[i][j]) * matrix_d_b2b_edge[i][j] * k_ball * (-delta_y);
            // Apply Newton's third law
            sf_b2b_all[2 * j][i] = -sf_b2b_all[2 * i][j];
            sf_b2b_all[2 * j + 1][i] = -sf_b2b_all[2 * i + 1][j];
        }
    }

    // Aggregate forces per agent
    Matrix sf_b2b(2, std::vector<double>(n_a));
    for (int i = 0; i < n_a; ++i) {
        for (int j = 0; j < dim; ++j) {
            double sum = 0.0;
            // Sum forces from all neighbors
            for (int k = 0; k < n_a; ++k) {
                sum += sf_b2b_all[2 * i + j][k];
            }
            sf_b2b[j][i] = sum;
        }
    }

    // Output to flattened array
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < n_a; ++j) {
           sf_b2b_input[i * n_a + j] = sf_b2b[i][j];
        }
    }
}

// Calculates distances to walls and collision status
void _get_dist_b2w(double *p_input, 
                   double *r_input, 
                   double *d_b2w_input, 
                   bool *isCollision_input, 
                   int dim, 
                   int n_a, 
                   double *boundary_pos) 
{
    // Initialize position matrix
    Matrix matrix_p(dim, std::vector<double>(n_a));
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < n_a; ++j) {
            matrix_p[i][j] = p_input[i * n_a + j];
        }
    }

    // Initialize distance and collision matrices
    Matrix d_b2w(4, std::vector<double>(n_a, 0.0));
    std::vector<std::vector<bool>> isCollision(4, std::vector<bool>(n_a, false));
    
    // Calculate distances to four boundaries (left, top, right, bottom)
    for (int i = 0; i < n_a; ++i) {
        d_b2w[0][i] = matrix_p[0][i] - r_input[i] - boundary_pos[0];  // Left boundary
        d_b2w[1][i] = boundary_pos[1] - (matrix_p[1][i] + r_input[i]);  // Top boundary
        d_b2w[2][i] = boundary_pos[2] - (matrix_p[0][i] + r_input[i]);  // Right boundary
        d_b2w[3][i] = matrix_p[1][i] - r_input[i] - boundary_pos[3];  // Bottom boundary
    }
    
    // Determine collisions and convert to absolute distances
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < n_a; ++j) {
            isCollision[i][j] = (d_b2w[i][j] < 0);
            d_b2w[i][j] = std::abs(d_b2w[i][j]);
        }
    }

    // Output to flattened arrays
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < n_a; ++j) {
            d_b2w_input[i * n_a + j] = d_b2w[i][j];
            isCollision_input[i * n_a + j] = isCollision[i][j];
        }
    }
}

// Concatenates two matrices along specified axis (0=rows, 1=columns)
Matrix _concatenate(const Matrix& arr1, const Matrix& arr2, int axis) {
    if (axis == 0) {  // Row-wise concatenation
        Matrix result(arr1.size() + arr2.size(), std::vector<double>(arr1[0].size()));
        // Copy first matrix
        for (size_t i = 0; i < arr1.size(); ++i) {
            std::copy(arr1[i].begin(), arr1[i].end(), result[i].begin());
        }
        // Append second matrix
        for (size_t i = 0; i < arr2.size(); ++i) {
            std::copy(arr2[i].begin(), arr2[i].end(), result[arr1.size() + i].begin());
        }
        return result;
    } else if (axis == 1) {  // Column-wise concatenation
        Matrix result(arr1.size(), std::vector<double>(arr1[0].size() + arr2[0].size()));
        // Combine columns
        for (size_t i = 0; i < arr1.size(); ++i) {
            std::copy(arr1[i].begin(), arr1[i].end(), result[i].begin());
            std::copy(arr2[i].begin(), arr2[i].end(), result[i].begin() + arr1[0].size());
        }
        return result;
    } else {
        return Matrix();  // Invalid axis
    }
}

// Extracts single column as a 2D matrix
Matrix _extract_column(const Matrix& arr, size_t col_index) {
    Matrix result;
    if (col_index < arr[0].size()) {
        for (const auto& row : arr) {
            result.push_back({row[col_index]});
        }
    }
    return result;
}

// Extracts single column as a 1D vector
std::vector<double> _extract_column_one(const Matrix& arr, size_t col_index) {
    std::vector<double> result;
    if (col_index < arr[0].size()) {
        for (const auto& row : arr) {
            result.push_back(row[col_index]);
        }
    }
    return result;
}

// Computes Euclidean norm of a vector
double _norm(std::vector<double>& v) {
    double sum = 0.0;
    for (double x : v) {
        sum += std::pow(x, 2);
    }
    return std::sqrt(sum);
}

// Checks if all elements in array are greater than (n_l - 1)
bool _all_elements_greater_than_(std::vector<int>& arr, int n_l) {
    for (int value : arr) {
        if (value <= (n_l - 1)) {
            return false;
        }
    }
    return true;
}

// Cosine decay function for smooth transitions
double _rho_cos_dec(double z, double delta, double r) {
    if (z < delta * r) {
        return 1.0;  // Full value in close range
    } else if (z < r) {
        // Cosine decay in transition zone
        return (1.0 / 2.0) * (1.0 + std::cos(M_PI * (z / r - delta) / (1.0 - delta)));
    } else {
        return 0.0;  // Zero beyond max range
    }
}

// Converts 1D vector to 2D column matrix
Matrix _vector_to_matrix(const std::vector<double>& vec) {
    Matrix matrix(vec.size(), std::vector<double>(1));
    for (size_t i = 0; i < vec.size(); ++i) {
        matrix[i][0] = vec[i];
    }
    return matrix;
}

// Flattens 2D matrix to 1D vector
std::vector<double> _matrix_to_vector(const Matrix& matrix) {
    std::vector<double> vec;
    for (const auto& row : matrix) {
        for (const auto& element : row) {
            vec.push_back(element);
        }
    }
    return vec;
}
