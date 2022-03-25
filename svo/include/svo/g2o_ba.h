#include <set>
#include <list>


#include "../Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"
#include "../Thirdparty/g2o/g2o/core/sparse_block_matrix.h"
#include "../Thirdparty/g2o/g2o/core/block_solver.h"
#include "../Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "../Thirdparty/g2o/g2o/core/optimization_algorithm_gauss_newton.h"
#include "../Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "../Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "../Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "../Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"

// g2o问题的食物链顶层是SparseOptimizer, 一个SparseOptimizer包含优化算法，
// g2o里面有三种优化算法，分别为 GN , LM , Dogleg,而每一种算法都需要执行Ax=b的求解，而求解这些矩阵就需要求解器Solver,用的最多的就是线性求解器，
// g2o里面有几种线性求解器分别为 cholmod ，　csparse，　dense，　eigen，　pcg;
// LinearSolverCholmod ：使用 sparse cholesky分解法。继承自LinearSolverCCS
// LinearSolverDense ：使用 dense cholesky 分解法。继承自LinearSolver
// LinearSolverCSparse：使用CSparse法。继承自LinearSolverCCS
// LinearSolverEigen： 依赖项只有eigen，使用eigen中 sparse Cholesky 求解，因此编译好后可以方便的在其他地方使用，性能和CSparse差不多。继承自LinearSolver
// LinearSolverPCG ：使用preconditioned conjugate gradient 法，继承自LinearSolver

#include <svo/common/frame.h>
#include "svo/global.h"

// namespace g2o {
//     class EdgeProjectXYZ2UV;
//     class SparseOptimizer;
//     class VertexSE3Expmap;
//     class VertexSBAPointXYZ;
// }
// typedef g2o::EdgeProjectXYZ2UV g2oEdgeSE3;
// typedef g2o::VertexSE3Expmap g2oFrameSE3;
// typedef g2o::VertexSBAPointXYZ g2oPoint;
// typedef g2o::SparseOptimizer sparseOptimizer;

namespace svo {
class G2oOptimizer {
 public:
    G2oOptimizer(){
        printf("creat g2o optimizer !!");
    }
    ~G2oOptimizer(){}

    void print() {
        printf("g2o optimizer !!");
    }
    /// Temporary container to hold the g2o edge with reference to frame and point.
//     struct EdgeContainerSE3 {
//         g2oEdgeSE3* edge;
//         svo::FramePtr frame;
//         svo::Feature* feature;
//         bool is_deleted;
//         EdgeContainerSE3(g2oEdgeSE3* e, svo::FramePtr frame, svo::Feature* feature)
//             : edge(e), frame(frame), feature(feature), is_deleted(false) {}
//     };

//     void localBA(svo::FramePtr center_kf,
//                 std::vector<svo::FramePtr>* core_kfs,
//                 size_t& n_incorrect_edges_1,
//                 size_t& n_incorrect_edges_2,
//                 double& init_error,
//                 double& final_error);

//  private:
//     void setupG2o(sparseOptimizer* optimizer);
//     void runSparseBAOptimizer(sparseOptimizer* optimizer,
//                             unsigned int num_iter,
//                             double& init_error,
//                             double& final_error);
//     g2oFrameSE3* createG2oFrameSE3(svo::FramePtr kf, size_t id, bool fixed);
//     g2oPoint* createG2oPoint(Eigen::Vector3d pos, size_t id, bool fixed);
//     g2oEdgeSE3* createG2oEdgeSE3(g2oFrameSE3* v_kf,
//                                 g2oPoint* v_mp,
//                                 const Eigen::Vector2d& f_up,
//                                 bool robust_kernel,
//                                 double huber_width,
//                                 double weight = 1);




void Optimizer::LocalBundleAdjustment(KeyFrame *pKF, bool* pbStopFlag, Map* pMap)
{    
    // Local KeyFrames: First Breath Search from Current Keyframe
    list<KeyFrame*> lLocalKeyFrames;
    lLocalKeyFrames.push_back(pKF);
    pKF->mnBALocalForKF = pKF->mnId;
    const vector<KeyFrame*> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
    // for(int i=0, iend=vNeighKFs.size(); i<iend; i++)
    // {
    //     KeyFrame* pKFi = vNeighKFs[i];
    //     pKFi->mnBALocalForKF = pKF->mnId;
    //     if(!pKFi->isBad())
    //         lLocalKeyFrames.push_back(pKFi);
    // }

    // Local MapPoints seen in Local KeyFrames
    list<MapPoint*> lLocalMapPoints;
    // for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin() , lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    // {
    //     vector<MapPoint*> vpMPs = (*lit)->GetMapPointMatches();
    //     for(vector<MapPoint*>::iterator vit=vpMPs.begin(), vend=vpMPs.end(); vit!=vend; vit++)
    //     {
    //         MapPoint* pMP = *vit;
    //         if(pMP)
    //             if(!pMP->isBad())
    //                 if(pMP->mnBALocalForKF!=pKF->mnId)
    //                 {
    //                     lLocalMapPoints.push_back(pMP);
    //                     pMP->mnBALocalForKF=pKF->mnId;
    //                 }
    //     }
    // }

    // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
    list<KeyFrame*> lFixedCameras;
    // for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    // {
    //     map<KeyFrame*,size_t> observations = (*lit)->GetObservations();
    //     for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
    //     {
    //         KeyFrame* pKFi = mit->first;
    //         if(pKFi->mnBALocalForKF!=pKF->mnId && pKFi->mnBAFixedForKF!=pKF->mnId)
    //         {                
    //             pKFi->mnBAFixedForKF=pKF->mnId;
    //             if(!pKFi->isBad())
    //                 lFixedCameras.push_back(pKFi);
    //         }
    //     }
    // }

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);
    if(pbStopFlag) { optimizer.setForceStopFlag(pbStopFlag);}

    unsigned long maxKFid = 0;
    // Set Local KeyFrame vertices
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++) {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(pKFi->mnId==0);
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;
    }

    // Set Fixed KeyFrame vertices
    for(list<KeyFrame*>::iterator lit=lFixedCameras.begin(), lend=lFixedCameras.end(); lit!=lend; lit++) {
        KeyFrame* pKFi = *lit;
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
        vSE3->setId(pKFi->mnId);
        vSE3->setFixed(true);
        optimizer.addVertex(vSE3);
        if(pKFi->mnId>maxKFid)
            maxKFid=pKFi->mnId;
    }

    // Set MapPoint vertices
    const int nExpectedSize = (lLocalKeyFrames.size()+lFixedCameras.size())*lLocalMapPoints.size();

    vector<g2o::EdgeSE3ProjectXYZ*> vpEdgesMono;
    vector<KeyFrame*> vpEdgeKFMono;
    vector<MapPoint*> vpMapPointEdgeMono;
    vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgesStereo;
    vector<KeyFrame*> vpEdgeKFStereo;
    vector<MapPoint*> vpMapPointEdgeStereo;
    vpEdgesMono.reserve(nExpectedSize);
    vpEdgeKFMono.reserve(nExpectedSize);
    vpMapPointEdgeMono.reserve(nExpectedSize);
    vpEdgesStereo.reserve(nExpectedSize);
    vpEdgeKFStereo.reserve(nExpectedSize);
    vpMapPointEdgeStereo.reserve(nExpectedSize);

    const float thHuberMono = sqrt(5.991);
    const float thHuberStereo = sqrt(7.815);

    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++) {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
        int id = pMP->mnId+maxKFid+1;
        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        const map<KeyFrame*,size_t> observations = pMP->GetObservations();

        //Set edges
        for(map<KeyFrame*,size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first;

            if(!pKFi->isBad())
            {                
                const cv::KeyPoint &kpUn = pKFi->mvKeysUn[mit->second];

                // Monocular observation
                if(pKFi->mvuRight[mit->second]<0)
                {
                    Eigen::Matrix<double,2,1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;

                    optimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vpEdgeKFMono.push_back(pKFi);
                    vpMapPointEdgeMono.push_back(pMP);
                }
                else // Stereo observation
                {
                    Eigen::Matrix<double,3,1> obs;
                    const float kp_ur = pKFi->mvuRight[mit->second];
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                    e->setInformation(Info);

                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberStereo);

                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;
                    e->bf = pKFi->mbf;

                    optimizer.addEdge(e);
                    vpEdgesStereo.push_back(e);
                    vpEdgeKFStereo.push_back(pKFi);
                    vpMapPointEdgeStereo.push_back(pMP);
                }
            }
        }
    }

    if(pbStopFlag) {
        if(*pbStopFlag) {return;}
    }

    optimizer.initializeOptimization();
    optimizer.optimize(5);

    bool bDoMore= true;
    if(pbStopFlag)
        if(*pbStopFlag)
            bDoMore = false;

    if(bDoMore) {
        // Check inlier observations
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
        {
            g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
            MapPoint* pMP = vpMapPointEdgeMono[i];

            if(pMP->isBad())
                continue;

            if(e->chi2()>5.991 || !e->isDepthPositive())
            {
                e->setLevel(1);
            }

            e->setRobustKernel(0);
        }

        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
        {
            g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
            MapPoint* pMP = vpMapPointEdgeStereo[i];

            if(pMP->isBad())
                continue;

            if(e->chi2()>7.815 || !e->isDepthPositive())
            {
                e->setLevel(1);
            }

            e->setRobustKernel(0);
        }

        // Optimize again without the outliers

        optimizer.initializeOptimization(0);
        optimizer.optimize(10);

    }

    vector<pair<KeyFrame*,MapPoint*> > vToErase;
    vToErase.reserve(vpEdgesMono.size()+vpEdgesStereo.size());

    // Check inlier observations       
    for(size_t i=0, iend=vpEdgesMono.size(); i<iend;i++)
    {
        g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
        MapPoint* pMP = vpMapPointEdgeMono[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>5.991 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFMono[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }

    for(size_t i=0, iend=vpEdgesStereo.size(); i<iend;i++)
    {
        g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
        MapPoint* pMP = vpMapPointEdgeStereo[i];

        if(pMP->isBad())
            continue;

        if(e->chi2()>7.815 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFStereo[i];
            vToErase.push_back(make_pair(pKFi,pMP));
        }
    }

    // Get Map Mutex
    unique_lock<mutex> lock(pMap->mMutexMapUpdate);

    if(!vToErase.empty())
    {
        for(size_t i=0;i<vToErase.size();i++)
        {
            KeyFrame* pKFi = vToErase[i].first;
            MapPoint* pMPi = vToErase[i].second;
            pKFi->EraseMapPointMatch(pMPi);
            pMPi->EraseObservation(pKFi);
        }
    }

    // Recover optimized data

    //Keyframes
    for(list<KeyFrame*>::iterator lit=lLocalKeyFrames.begin(), lend=lLocalKeyFrames.end(); lit!=lend; lit++)
    {
        KeyFrame* pKF = *lit;
        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->mnId));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        pKF->SetPose(Converter::toCvMat(SE3quat));
    }
    //Points
    for(list<MapPoint*>::iterator lit=lLocalMapPoints.begin(), lend=lLocalMapPoints.end(); lit!=lend; lit++)
    {
        MapPoint* pMP = *lit;
        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->mnId+maxKFid+1));
        pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
        pMP->UpdateNormalAndDepth();
    }
}





};

}  // namespace svo

