from callback import *
from easypbr import Scene

class ViewerCallback(Callback):

    def __init__(self):
        pass

    def after_forward_pass(self, pred_softmax, cloud, **kwargs):
        # print("AFTER FORWARD PASS")

        mesh_pred=cloud.clone()
        l_pred=pred_softmax.detach().argmax(axis=1).cpu().numpy()
        mesh_pred.L_pred=l_pred
        mesh_pred.m_vis.m_point_size=4
        mesh_pred.m_vis.set_color_semanticpred()
        # mesh_pred.move_in_z(cloud.get_scale()) 
        Scene.show(mesh_pred, "mesh_pred")
