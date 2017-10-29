// Copyright (c) 2016 University of Minnesota
//
// MPM-OPTIMIZATION Uses the BSD 2-Clause License (http://www.opensource.org/licenses/BSD-2-Clause)
// Redistribution and use in source and binary forms, with or without modification, are
// permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice, this list of
// conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list
// of conditions and the following disclaimer in the documentation and/or other materials
// provided with the distribution.
// THIS SOFTWARE IS PROVIDED "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR  A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY OF MINNESOTA, DULUTH OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
// OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
// IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
// OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// By Matt Overby (http://www.mattoverby.net)

#include "MPM.hpp"

using namespace mpm;

//
//	Declare static members
//	These are default, usually overwritten by the solver.
//
Vec3r MPM::cellsize(0.25, 0.25, 0.25);
Vec3i MPM::gridsize(32, 32, 32);
Real  MPM::timestep_s = 0.04;

// Pass in (normalized) grid index and returns its index in the grid vector.
int MPM::grid_index(const Vec3i& pos)
{
    return (pos[2] * gridsize[2] * gridsize[1]) + (pos[1] * gridsize[0]) + pos[0];
}

// Inverse of grid_index (position in grid, NOT world space!)
Vec3i MPM::grid_pos(int grid_idx)
{
    int z = grid_idx / (gridsize[0] * gridsize[1]);
    grid_idx -= (z * gridsize[0] * gridsize[1]);
    int y = grid_idx / gridsize[0];
    int x = grid_idx % gridsize[0];
    return Vec3i(x, y, z);
}

//
//	Clear the grid
//
void MPM::clear_grid(std::vector<GridNode*>& active_grid)
{
        #pragma omp parallel for
    for(int i = 0; i < active_grid.size(); ++i) {
        GridNode* g = active_grid[i];

        g->v          = Vec3r(0, 0, 0);
        g->m          = 0.0;
        g->active_idx = 0;

        g->particles.clear();
        g->wip.clear();
        g->dwip.clear();
    }

    active_grid.clear();
} // end clear grid

//
//	Particle to grid: masses and interpolation weights
//
void MPM::p2g_mass(std::vector<Particle*>& particles, std::vector<GridNode*>& grid, std::vector<GridNode*>& active_grid)
{
        #pragma omp parallel for
    for(int i = 0; i < particles.size(); ++i) {
        Particle* p = particles[i];
        p->D = Mat3x3r(0);

        std::vector<int>   grid_nodes;
        std::vector<Real>  wip;
        std::vector<Vec3r> dwip;
        MPM::make_grid_loop(p, grid_nodes, wip, dwip);
        for(int j = 0; j < grid_nodes.size(); ++j) {
            // Compute helper matrix
            Vec3r xixp = xi_xp(grid_nodes[j], p);
            p->D += wip[j] * glm::outerProduct(xixp, xixp);             // eq 174 course notes

            // Atomics are bad and all, but this way works fine for now.
                        #pragma omp critical
            {             // add mass to grid
                int grid_idx = grid_nodes[j];
                grid[ grid_idx ]->m += wip[j] * p->m;
                grid[ grid_idx ]->particles.push_back(p);
                grid[ grid_idx ]->wip.push_back(wip[j]);
                grid[ grid_idx ]->dwip.push_back(dwip[j]);
            }
        } // end loop grid nodes
    }     // end loop particles

    // Create the active grid
    for(int i = 0; i < grid.size(); ++i) {
        if(grid[i]->m > 0.0) {
            grid[i]->active_idx = active_grid.size();
            active_grid.push_back(grid[i]);
        }
    }
} // end p2g mass and weights

//
//	Calculate (initial) particle volumes
//
void MPM::init_volumes(std::vector<Particle*>& particles, const std::vector<GridNode*>& grid)
{
        #pragma omp parallel for
    for(int i = 0; i < particles.size(); ++i) {
        Particle* p    = particles[i];
        Real      dens = 0.0;

        std::vector<int>   grid_nodes;
        std::vector<Real>  wip;
        std::vector<Vec3r> dwip;
        MPM::make_grid_loop(p, grid_nodes, wip, dwip);
        for(int j = 0; j < grid_nodes.size(); ++j) {
            dens += wip[j] * grid[ grid_nodes[j] ]->m;
        }

        dens /= (cellsize[0] * cellsize[1] * cellsize[2]); // density
        assert(dens > 0.0);
        p->vol = p->m / dens;                              // volume
    }                                                      // end loop particles
} // end compute particle volumes

//
//	Particle to Grid Velocity
//
void MPM::p2g_velocity(const std::vector<Particle*>& particles, std::vector<GridNode*>& active_grid)
{
        #pragma omp parallel for
    for(int i = 0; i < active_grid.size(); ++i) {
        Vec3i gridx = MPM::grid_pos(active_grid[i]->global_idx);

        for(int j = 0; j < active_grid[i]->particles.size(); ++j) {
            Particle* p      = active_grid[i]->particles[j];
            Mat3x3r   Dp_inv = glm::inverse(p->D);

            Vec3r xixp(gridx[0] * cellsize[0] - p->x[0], gridx[1] * cellsize[1] - p->x[1], gridx[2] * cellsize[2] - p->x[2]);
            Vec3r newv = p->v + p->B * Dp_inv * (xixp);
            active_grid[i]->v += active_grid[i]->wip[j] * p->m * newv; // eq 173 course notes
        }                                                              // end loop neighborhood

        // Fix Velocity (remove mass)
        assert(active_grid[i]->m > 0.0);
        active_grid[i]->v /= active_grid[i]->m;
    }     // end loop grid
} // end p2g velocity

//
//	Update the particle deformation gradient (step 6 course notes)
//
void MPM::g2p_deformation(std::vector<Particle*>& particles, const std::vector<GridNode*>& grid)
{
        #pragma omp parallel for
    for(int i = 0; i < particles.size(); ++i) {
        Particle* p = particles[i];

        Mat3x3r velocity_grad(0);

        std::vector<int>   grid_nodes;
        std::vector<Real>  wip;
        std::vector<Vec3r> dwip;
        MPM::make_grid_loop(p, grid_nodes, wip, dwip);
        for(int j = 0; j < grid_nodes.size(); ++j) {
            velocity_grad += glm::outerProduct(grid[ grid_nodes[j] ]->v, dwip[j]);
        }

        p->update_deform_grad(velocity_grad, timestep_s); // eq 181 course notes
    }                                                     // end loop particles
} // end update deformation gradient

//
//	Grid to Particle velocity and affine matrix
//
void MPM::g2p_velocity(std::vector<Particle*>& particles, const std::vector<GridNode*>& grid)
{
    //
    //	Compute particle velocities and affine matrix
    //
        #pragma omp parallel for
    for(int i = 0; i < particles.size(); ++i) {
        Particle* p = particles[i];
        p->v = Vec3r(0);
        p->B = Mat3x3r(0);

        std::vector<int>   grid_nodes;
        std::vector<Real>  wip;
        std::vector<Vec3r> dwip;
        MPM::make_grid_loop(p, grid_nodes, wip, dwip);

        for(int j = 0; j < grid_nodes.size(); ++j) {
            p->v += wip[j] * grid[grid_nodes[j]]->v;                          // eq 175 course notes
            Vec3r xixp = xi_xp(grid_nodes[j], p);
            p->B += wip[j] * glm::outerProduct(grid[grid_nodes[j]]->v, xixp); // eq 176 course notes
        }
    }                                                                         // end loop particles
} // end grid to particle velocity

//
//	Particle Advection
//
void MPM::particle_advection(std::vector<Particle*>& particles)
{
        #pragma omp parallel for
    for(int i = 0; i < particles.size(); ++i) {
        particles[i]->x += timestep_s * particles[i]->v; // assumes APIC
    }                                                    // end loop particles
} // end particle advection

//
//	Grid Collision
//
void MPM::grid_collision(std::vector<GridNode*>& active_grid)
{
        #pragma omp parallel for
    for(int i = 0; i < active_grid.size(); ++i) {
        GridNode* node  = active_grid[i];
        Vec3i     gridx = grid_pos(node->global_idx);

        // Boundary:
        for(int j = 0; j < 3; ++j) {
            if(gridx[j] < 2) { node->v[j] = 0.f; }
            if(gridx[j] > gridsize[j] - 2) { node->v[j] = 0.f; }
        }
    }     // end loop active grid
} // end check grid collision

//
//	Interpolation Functions
//


void MPM::make_grid_loop(const Particle* p, std::vector<int>& grid_idx, std::vector<Real>& Wip, std::vector<Vec3r>& dWip)
{
    grid_idx.clear(); Wip.clear(); dWip.clear();
    grid_idx.reserve(64);
    Wip.reserve(64);
    dWip.reserve(64);

    // Our smoothing kernel has a radius of 2: 64 nodes 3D
    Vec3i normed_x = Vec3i(p->x[0] / cellsize[0], p->x[1] / cellsize[1], p->x[2] / cellsize[2]);
    Vec3i ll_idx   = normed_x - Vec3i(1, 1, 1); // get the lowest-left index

    for(int x = ll_idx[0]; x < (ll_idx[0] + 4); ++x) {
        Real wx_x   = p->x[0] / cellsize[0] - Real(x);
        Real wip_x  = qspline(wx_x);
        Real dwip_x = d_qspline(wx_x);

        for(int y = ll_idx[1]; y < (ll_idx[1] + 4); ++y) {
            Real wy_x   = p->x[1] / cellsize[1] - Real(y);
            Real wip_y  = qspline(wy_x);
            Real dwip_y = d_qspline(wy_x);

            for(int z = ll_idx[2]; z < (ll_idx[2] + 4); ++z) {
                Real wz_x   = p->x[2] / cellsize[2] - Real(z);
                Real wip_z  = qspline(wz_x);
                Real dwip_z = d_qspline(wz_x);

                // Compute grid index and interpolation weights
                int   g_idx = grid_index(Vec3i(x, y, z));
                Real  g_wip = wip_x * wip_y * wip_z;
                Vec3r g_dwip(
                    (1.0 / cellsize[0]) * dwip_x* wip_y* wip_z,
                    wip_x * (1.0 / cellsize[1]) * dwip_y* wip_z,
                    wip_x* wip_y * (1.0 / cellsize[2]) * dwip_z
                    );             // eq (after) 124 course notes

                // Check that the computed node is within the grid boundaries.
                // Near the edges it may not be (if something went wrong with collision).
                if(g_idx < 0 || g_idx > gridsize[0] * gridsize[1] * gridsize[2] - 1 || g_wip < 0.0) { continue; }

                // All good, add it to the vectors
                grid_idx.push_back(g_idx);
                Wip.push_back(g_wip);
                dWip.push_back(g_dwip);
            } // end loop z
        }     // end loop y
    }         // end loop x
} // end make grid loop

Vec3r MPM::xi_xp(const int grid_idx, const Particle* p)
{
    Vec3i gridx = grid_pos(grid_idx);
    Vec3r xixp(Real(gridx[0]) * cellsize[0] - p->x[0], Real(gridx[1]) * cellsize[1] - p->x[1], Real(gridx[2]) * cellsize[2] - p->x[2]);
    return xixp;
} // end compute world coord diff
