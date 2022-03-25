

#include "svo/g2o_ba.h"

#include <vikit/math_utils.h>
#include <boost/thread.hpp>

#include <svo/global.h>
#include <svo/map.h>
#include <set>
#include <list>

#include <svo/common/frame.h>

#define SCHUR_TRICK 1

namespace svo {

void G2oOptimizer::localBA(svo::FramePtr center_kf,
             std::vector<svo::FramePtr>* core_kfs,
             size_t& n_incorrect_edges_1,
             size_t& n_incorrect_edges_2,
             double& init_error,
             double& final_error) {
    g2o::SparseOptimizer optimizer;
    {
        optimizer.setVerbose(false);
        // solver
        g2o::BlockSolver_6_3::LinearSolverType* linearSolver;
        // linearSolver = new g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>();
        linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
        // linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();
        g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        solver->setMaxTrialsAfterFailure(5);
        optimizer.setAlgorithm(solver);

        // g2o::BlockSolverX::LinearSolverType* linearSolver;
        // linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();
        // g2o::BlockSolverX* solver_ptr = new g2o::BlockSolverX(linearSolver);
        // g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        // solver->setUserLambdaInit(1e0);
        // optimizer.setAlgorithm(solver);

        // setup camera
        g2o::CameraParameters* cam_params = new g2o::CameraParameters(1.0, Eigen::Vector2d(0., 0.), 0.);
        cam_params->setId(0);
        if (!optimizer.addParameter(cam_params)) { assert(false);}
        // setupG2o(&optimizer);
    }
    std::list<EdgeContainerSE3> edges;
    std::set<svo::PointPtr> mps;
    std::list<svo::FramePtr> neib_kfs;
    size_t v_id = 0;
    size_t n_mps = 0;
    size_t n_fix_kfs = 0;
    size_t n_var_kfs = 1;
    size_t n_edges = 0;
    n_incorrect_edges_1 = 0;
    n_incorrect_edges_2 = 0;
    // Add all core keyframes
    for (auto frame : (*core_kfs)) {
        g2oFrameSE3* v_kf = createG2oFrameSE3(frame, v_id++, false);
        // frame->v_kf_ = v_kf;
        ++n_var_kfs;
        assert(optimizer.addVertex(v_kf));
        // all points that the core keyframes observe are also optimized:
        for(size_t i = 0; i < frame->num_features_; ++i) {
            if (frame->landmark_vec_[i] == nullptr || !isCorner(frame->type_vec_[i])) {
                continue;
            }
            mps.insert(frame->landmark_vec_.at(i));
        }
    }
    double lobaThresh = 2.0;
    double poseOptimThresh = 2.0;
    double reproj_thresh_2 = lobaThresh/ center_kf->getErrorMultiplier();
    double reproj_thresh_1 = poseOptimThresh / center_kf->getErrorMultiplier();
    double reproj_thresh_1_squared = reproj_thresh_1 * reproj_thresh_1;
    for (std::set<svo::PointPtr>::iterator it_pt = mps.begin(); it_pt != mps.end(); ++it_pt) {
        // Create point vertex
        g2oPoint* v_pt = createG2oPoint((*it_pt)->pos_, v_id++, true);
        // (*it_pt)->v_pt_ = v_pt;
        assert(optimizer.addVertex(v_pt));
        ++n_mps;
        // Add edges
        auto it_obs = (*it_pt)->obs_.begin();
        while (it_obs != (*it_pt)->obs_.end()) {
            svo::FramePtr frame = (*it_obs).frame.lock();
            // if (frame->v_kf_ == NULL) {
            //     g2oFrameSE3* v_kf = createG2oFrameSE3(frame, v_id++, true);
            //     frame->v_kf_ = v_kf;
            //     ++n_fix_kfs;
            //     assert(optimizer.addVertex(v_kf));
            //     neib_kfs.push_back(frame);
            // }
            // // undistort point
            // Eigen::Vector2d obs = vk::project2(frame->f_vec_.col(it_obs->keypoint_index_));
            // g2oEdgeSE3* e = createG2oEdgeSE3(
            //     frame->v_kf_, v_pt, obs, true, reproj_thresh_2, 1.0 / (1 << frame->level_vec_[it_obs->keypoint_index_]));
            // assert(optimizer.addEdge(e));
            // edges.push_back(EdgeContainerSE3(e, frame, nullptr));
            ++n_edges;
            ++it_obs;
        }
    }

    // structure only
    g2o::StructureOnlySolver<3> structure_only_ba;
    g2o::OptimizableGraph::VertexContainer points;
    for (g2o::OptimizableGraph::VertexIDMap::const_iterator it = optimizer.vertices().begin(); it != optimizer.vertices().end(); ++it) {
        g2o::OptimizableGraph::Vertex* v = static_cast<g2o::OptimizableGraph::Vertex*>(it->second);
        if (v->dimension() == 3 && v->edges().size() >= 2) points.push_back(v);
    }
    structure_only_ba.calc(points, 10);
    // Optimization
    int iter_times = 4;
    if (iter_times > 0) {
        runSparseBAOptimizer(&optimizer, iter_times, init_error, final_error);
    }
    // Update :
    // for (auto it = core_kfs->begin(); it != core_kfs->end(); ++it) {
    //     // (*it)->T_f_w_ = SE3((*it)->v_kf_->estimate().rotation(),
    //     //                     (*it)->v_kf_->estimate().translation());
    //     (*it)->v_kf_ = NULL;
    // }
    // for (auto it = neib_kfs.begin(); it != neib_kfs.end(); ++it) {
    //     (*it)->v_kf_ = NULL;
    // }
    // Update Mappoints
    for (auto it = mps.begin(); it != mps.end(); ++it) {
        // (*it)->pos_ = (*it)->v_pt_->estimate();
        // (*it)->v_pt_ = NULL;
    }

    // Remove Measurements with too large reprojection error
    double reproj_thresh_2_squared = reproj_thresh_2 * reproj_thresh_2;
    for (std::list<EdgeContainerSE3>::iterator it = edges.begin(); it != edges.end(); ++it) {
        if (it->edge->chi2() > reproj_thresh_2_squared) {  //*(1<<it->feature_->level))
            // map->removePtFrameRef(it->frame, it->feature);
            ++n_incorrect_edges_2;
        }
    }
    // TODO: delete points and edges!
    init_error = sqrt(init_error) * center_kf->getErrorMultiplier();
    final_error = sqrt(final_error) * center_kf->getErrorMultiplier();
}

// void G2oOptimizer::setupG2o(sparseOptimizer* optimizer) {
// //     optimizer->setVerbose(false);

// // #if SCHUR_TRICK
// //     // solver
// //     g2o::BlockSolver_6_3::LinearSolverType* linearSolver;
// //     linearSolver =
// //         new g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>();
// //     // linearSolver = new
// //     // g2o::LinearSolverCSparse<g2o::BlockSolver_6_3::PoseMatrixType>();
// //     g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
// //     g2o::OptimizationAlgorithmLevenberg* solver =
// //         new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
// // #else
// //     g2o::BlockSolverX::LinearSolverType* linearSolver;
// //     linearSolver =
// //         new g2o::LinearSolverCholmod<g2o::BlockSolverX::PoseMatrixType>();
// //     // linearSolver = new
// //     // g2o::LinearSolverCSparse<g2o::BlockSolverX::PoseMatrixType>();
// //     g2o::BlockSolverX* solver_ptr = new g2o::BlockSolverX(linearSolver);
// //     g2o::OptimizationAlgorithmLevenberg* solver =
// //         new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
// // #endif

// //     solver->setMaxTrialsAfterFailure(5);
// //     optimizer->setAlgorithm(solver);

// //     // setup camera
// //     g2o::CameraParameters* cam_params =
// //         new g2o::CameraParameters(1.0, Eigen::Vector2d(0., 0.), 0.);
// //     cam_params->setId(0);
// //     if (!optimizer->addParameter(cam_params)) {
// //         assert(false);
// //     }
// }

// void G2oOptimizer::runSparseBAOptimizer(sparseOptimizer* optimizer,
//                           unsigned int num_iter,
//                           double& init_error,
//                           double& final_error) {
//     optimizer->initializeOptimization();
//     optimizer->computeActiveErrors();
//     init_error = optimizer->activeChi2();
//     optimizer->optimize(num_iter);
//     final_error = optimizer->activeChi2();
// }

// g2oFrameSE3* G2oOptimizer::createG2oFrameSE3(svo::FramePtr frame, size_t id, bool fixed) {
//     g2oFrameSE3* v = new g2oFrameSE3();
//     v->setId(id);
//     v->setFixed(fixed);
//     Eigen::Quaterniond q = frame->T_f_w_.getRotation().toImplementation();
//     q.normalize();
//     v->setEstimate(g2o::SE3Quat(q, frame->T_f_w_.getPosition()));
//     return v;
// }

// g2oPoint* G2oOptimizer::createG2oPoint(Eigen::Vector3d pos, size_t id, bool fixed) {
//     g2oPoint* v = new g2oPoint();
//     v->setId(id);
// #if SCHUR_TRICK
//     v->setMarginalized(true);
// #endif
//     v->setFixed(fixed);
//     v->setEstimate(pos);
//     return v;
// }

// g2oEdgeSE3* G2oOptimizer::createG2oEdgeSE3(g2oFrameSE3* v_frame,
//                              g2oPoint* v_point,
//                              const Eigen::Vector2d& f_up,
//                              bool robust_kernel,
//                              double huber_width,
//                              double weight) {
//     g2oEdgeSE3* e = new g2oEdgeSE3();
//     e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(v_point));
//     e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(v_frame));
//     e->setMeasurement(f_up);
//     e->information() = weight * Eigen::Matrix2d::Identity(2, 2);
//     g2o::RobustKernelHuber* rk =
//         new g2o::RobustKernelHuber();  // TODO: memory leak
//     rk->setDelta(huber_width);
//     e->setRobustKernel(rk);
//     e->setParameterId(0, 0);  // old: e->setId(v_point->id());
//     return e;
// }

}  // namespace ba
// }  // namespace svo
