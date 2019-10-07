

import rospy
import numpy as np
import sensor_msgs.msg as senmsg
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ros_numpy
import pcl
import geometry_msgs.msg as geomsg
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TwistStamped
import math
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
from scipy.spatial import Voronoi
import threading





class TopicSubscriber():
    def __init__(self):
        self.p=None
        self.merge=None
        self.vertices_hulls=[]
    """
    def cross(o, a, b): 
        return (a[0] - o[0]) * (b[1] - o[1]) -\
            (a[1] - o[1]) * (b[0] - o[0])
    """


    def pointcloudcallback(self,data):
        self.pc = ros_numpy.numpify(data)
        self.points=np.zeros((self.pc.shape[0],3))
        self.points[:,0]=self.pc['x']
        self.points[:,1]=self.pc['y']
        self.points[:,2]=self.pc['z']

        self.p=self.points
        z_mask1=self.p[:,2]>-1.6
        z_mask2=self.p[:,2]<-0.5
        z_mask=np.logical_and(z_mask1,z_mask2)
        
        self.p_zmasked=self.p[z_mask]
        x_mask1=self.p_zmasked[:,0]>-1.5
        x_mask2=self.p_zmasked[:,0]<25
        x_mask=np.logical_and(x_mask1,x_mask2)
        
        self.p_xzmasked=self.p_zmasked[x_mask]
        
        scaler=StandardScaler()
        X_scaled=scaler.fit_transform(self.p_xzmasked[:,0:2])
        dbscan=DBSCAN(eps=0.5,min_samples=5).fit(self.p_xzmasked[:,0:2])
        self.clusters=dbscan.fit_predict(X_scaled)
        self.cl=self.clusters
        self.clusters=np.array(self.clusters).reshape(-1,1)

       
        self.merge=np.concatenate((self.p_xzmasked,self.clusters),axis=1)

        self.points_2d=self.merge[:,0:2]
        
        hull_list=[]
        for i in range(self.cl.min(),self.cl.max()+1):
            a=np.where(self.merge[:,3]==i)
            hull_list.append(self.points_2d[a])

        self.hull=[]
        for j in range (len(set(self.cl))):
            if len(hull_list[j])>=3:
                self.hull.append(ConvexHull(hull_list[j]))

        vertices=[]
        for f in range (len (self.hull)):
            b=self.hull[f].points[self.hull[f].vertices]
            vertices.append(b)

        self.vertices=np.array(vertices)
        self.vertices_hulls=np.concatenate(self.vertices)

        self.tri=Delaunay(self.vertices_hulls)
        vor=Voronoi(self.vertices_hulls)

        mask=self.tri.find_simplex(vor.vertices)!=-1    #kordináták amik a delauney háromszögön belül vannak
        self.vertices_voronoi=vor.vertices[mask]
        """
        lower = []
        for e in range(len(self.hull)):
            for pp in self.hull[e].points:
                while len(lower) >= 2 and TopicSubscriber.cross(lower[-2], lower[-1], pp)\
                        <= 0:
                    lower.pop()
                lower.append(pp)

        upper = []
        for e in range(len(self.hull)):
            for pp in reversed(self.hull[e].points):
                while len(upper) >= 2 and TopicSubscriber.cross(upper[-2], upper[-1], pp) \
                        <= 0:
                    upper.pop()
                upper.append(pp)
        self.upper=np.array(upper)
        self.lower=np.array(lower)
        """
        
        mask1=[]
        for k in vor.ridge_vertices:
            k=np.asarray(k)
            c=np.all(k>=0)
            mask1.append(c)                     #csak a folytonos vonalakhoz tartozó indexek

        ridges=vor.ridge_vertices
        ridges=np.array(ridges)
        ridges=ridges[mask1]

        ridge_points=vor.ridge_points
        ridge_points=np.array(ridge_points)
        ridge_points=ridge_points[mask1]        #folytonos vonalakhoz tartozó pontok indexei     

        self.vor_vertices_ridges=vor.vertices[ridges]
        ins=self.tri.find_simplex(self.vor_vertices_ridges)

        mask2=[]
        for g in range(len(ins)):
            gg=np.all(ins[g]!=-1)              #azok amik a delauney háromszögön belül vannak
            mask2.append(gg)

        self.vor_vertices_ridges=self.vor_vertices_ridges[mask2]
        ridge_points=ridge_points[mask2]

        delauney_vertices=self.vertices_hulls[self.tri.vertices]

        list1=[]
        for aaa in range(self.vor_vertices_ridges.shape[0]):
            for fff in range(2):
                bbb=sum(sum(sum(self.vor_vertices_ridges[aaa,fff]==self.vor_vertices_ridges)))/2
                list1.append(bbb)

        list1=np.array(list1)
        index2=np.where(list1==1)
        index2=np.array(index2).reshape(-1)
        matrix=np.zeros(((len(index2)),))


        for jjj in range(len(index2)):
            matrix[jjj]=index2[jjj]//2

        matrix=matrix.astype(int)

        self.vor_vertices_ridges=np.delete(self.vor_vertices_ridges,matrix,0)

        self.p_voronoi_vertices=np.polyfit(self.vertices_voronoi[:,0],self.vertices_voronoi[:,1],3)

        m=np.rad2deg((self.vor_vertices_ridges[:,1,1]-self.vor_vertices_ridges[:,0,1])/(self.vor_vertices_ridges[:,1,0]-self.vor_vertices_ridges[:,0,1]))
        
        print(m)
        
        

    def current_velocitycallback(self,msg):
        self.lin_x=msg.twist.linear.x
        self.lin_y=msg.twist.linear.y
        self.ang_z=msg.twist.angular.z

        zero=[0,0]
        self.x=(math.cos(self.ang_z)*self.lin_x)
        self.y=(math.sin(self.ang_z)*self.lin_x)
        self.vector=[[0,self.x],[0,self.y]]
        self.vel_slope=np.rad2deg(self.y/self.x)

        #"""
        #fig = plt.figure()
        #plt.triplot(self.vertices_hulls[:,0], self.vertices_hulls[:,1], self.tri.simplices.copy())  #delauney háromszögelés
        plt.scatter(self.vertices_voronoi[:,0],self.vertices_voronoi[:,1],c='g')                #szűrt voronoi középpontok
        #plt.scatter(zero[0],zero[1],c='orange')                                       #origó
        #plt.scatter(self.p[:,0],self.p[:,1],c='orange')                                        #eredeti pontok
        plt.scatter(self.vertices_hulls[:,0],self.vertices_hulls[:,1],c="r")                   #héj pontok
        #plt.scatter(self.vertices_voronoi[:,0],np.polyval(self.p_voronoi_vertices,self.vertices_voronoi[:,0]),c='r')  #polinom illesztés voronoi kkpontokra
        #plt.plot(self.upper[:,0],self.upper[:,1],c="b")
        #plt.plot(self.lower[:,0],self.lower[:,1],c='r')
        
        for e in range(len(self.hull)):
            #plt.scatter(p[:,0],p[:,1],c=cl)                                                               #klaszterezett eredeti pontok
            plt.plot(self.hull[e].points[self.hull[e].vertices,0], self.hull[e].points[self.hull[e].vertices,1], 'k--', lw=2)  #konvex héj kirajzolás
        

        for index in range(len(self.vor_vertices_ridges)):
            plt.plot(self.vor_vertices_ridges[index,:,0],self.vor_vertices_ridges[index,:,1],c='k')                  #Voronoi kpontok/ref jel
        
    
        plt.plot(self.vector[0],self.vector[1],c='b')
        plt.scatter(zero[0],zero[1],c='m')
        #plt.scatter(self.p_xzmasked[:,0],self.p_xzmasked[:,1],c=self.cl)
        #plt.scatter(self.points_2d[:,0],self.points_2d[:,1])
        
        plt.show()
        #"""    
        
        print(self.vel_slope)

def callback():
    Ts=TopicSubscriber()    

    rospy.init_node("listener", anonymous=True)
    rospy.Subscriber("/points_raw", senmsg.PointCloud2,Ts.pointcloudcallback)
    rospy.Subscriber("/current_velocity", geomsg.TwistStamped,Ts.current_velocitycallback)
    rospy.spin()


if __name__ == '__main__':
    try:
        callback()
        
    except rospy.ROSInterruptException:
        pass