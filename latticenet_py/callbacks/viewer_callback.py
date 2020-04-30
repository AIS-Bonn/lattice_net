from callbacks.callback import *
from easypbr import Scene
import numpy as np

class ViewerCallback(Callback):

    def __init__(self):
        pass

    def after_forward_pass(self, pred_softmax, cloud, **kwargs):
        self.show_predicted_cloud(pred_softmax, cloud)
        self.show_difference_cloud(pred_softmax, cloud)


    def show_predicted_cloud(self, pred_softmax, cloud):
        mesh_pred=cloud.clone()
        l_pred=pred_softmax.detach().argmax(axis=1).cpu().numpy()
        mesh_pred.L_pred=l_pred
        mesh_pred.m_vis.m_point_size=4
        mesh_pred.m_vis.set_color_semanticpred()
        # mesh_pred.move_in_z(cloud.get_scale()) 
        Scene.show(mesh_pred, "mesh_pred")

    def show_difference_cloud(self,pred_softmax, cloud):
        mesh_pred=cloud.clone()
        l_pred=pred_softmax.detach().argmax(axis=1).cpu().numpy() #(nr_points, )
        l_gt=cloud.L_gt  #(nr_points,1)
        l_pred=np.expand_dims(l_pred,1) #(nr_points,1)

        diff=l_pred!=cloud.L_gt
        diff_repeated=np.repeat(diff, 3,1) #repeat 3 times on axis 1 so to obtain a (nr_points,3)

        mesh_pred.C=diff_repeated
        mesh_pred.m_vis.m_point_size=4
        mesh_pred.m_vis.set_color_pervertcolor()
        # mesh_pred.move_in_z(-cloud.get_scale()) #good for shapenetpartseg
        mesh_pred.translate_model_matrix([0.0, 0.0, -2.0]) #good for shapenetpartseg
        Scene.show(mesh_pred, "mesh_diff")

    def show_confidence_cloud(self, pred_softmax, cloud):
        mesh_pred=cloud.clone()
        l_pred_confidence,_=pred_softmax.detach().exp().max(axis=1)
        l_pred_confidence=l_pred_confidence.cpu().numpy() #(nr_points, )
        l_pred_confidence=np.expand_dims(l_pred_confidence,1) #(nr_points,1)
        l_pred_confidence=np.repeat(l_pred_confidence, 3,1) #repeat 3 times on axis 1 so to obtain a (nr_points,3)

        mesh_pred.C=l_pred_confidence
        mesh_pred.m_vis.m_point_size=4
        mesh_pred.m_vis.set_color_pervertcolor()
        # mesh_pred.move_in_z(-cloud.get_scale()) #good for shapenetpartseg
        # mesh_pred.move_in_z(-1.0) #good for shapenetpartseg
        Scene.show(mesh_pred, "mesh_confidence")


    def show_pca_of_features_cloud(self, per_point_features, cloud):
        mesh=cloud.clone()


        ## http://agnesmustar.com/2017/11/01/principal-component-analysis-pca-implemented-pytorch
        X=per_point_features.detach().squeeze(0).cpu()#we switch to cpu because svd for gpu needs magma: No CUDA implementation of 'gesdd'. Install MAGMA and rebuild cutorch (http://icl.cs.utk.edu/magma/) at /opt/pytorch/aten/src/THC/generic/THCTensorMathMagma.cu:191
        k=3
        # print("x is ", X.shape)
        X_mean = torch.mean(X,0)
        # print("x_mean is ", X_mean.shape)
        X = X - X_mean.expand_as(X)

        U,S,V = torch.svd(torch.t(X)) 
        C = torch.mm(X,U[:,:k])
        # print("C has shape ", C.shape)
        # print("C min and max is ", C.min(), " ", C.max() )
        C-=C.min()
        C/=C.max()
        # print("after normalization C min and max is ", C.min(), " ", C.max() )


        mesh.C=C.detach().cpu().numpy()
        mesh.m_vis.m_point_size=10
        mesh.m_vis.set_color_pervertcolor()
        mesh.move_in_z(-2*cloud.get_scale()) #good for shapenetpartseg
        Scene.show(mesh, "mesh_pca")
