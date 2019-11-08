#!/usr/bin/env python


import rospy
import numpy as np
import sensor_msgs.msg as senmsg
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
import matplotlib.pyplot as plt
import ros_numpy
import geometry_msgs.msg as geomsg
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TwistStamped
import math
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
from scipy.spatial import Voronoi
from sklearn import linear_model
from scipy import stats
from shapely.geometry import Point, Polygon
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from shapely.geometry import LineString
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

front_left_distance_from_Lidar=(1.869,0.763) 
rear_left_distance_from_Lidar=(-1.844,0.763)
rear_right_distance_from_Lidar=(-1.844,-0.7745)
front_right_distance_from_Lidar=(1.869,-0.7745)



class TopicSubscriber():
    global front_left_distance_from_Lidar,rear_left_distance_from_Lidar,rear_right_distance_from_Lidar,front_right_distance_from_Lidar
    def __init__(self):
        self.points=None
        self.vertices_voronoi=None
        self.vertices_XY=[]
        
        self.velodyne_sub=rospy.Subscriber("/velodyne_points", senmsg.PointCloud2,self.pointcloudcallback)
        self.velocity_sub=rospy.Subscriber("/current_velocity", geomsg.TwistStamped,self.current_velocitycallback)
        self.merge=None
        self.vertices_hulls=[]
    
    def cross(self,o, a, b): 
        return (a[0] - o[0]) * (b[1] - o[1]) -\
            (a[1] - o[1]) * (b[0] - o[0])

    def rotate(self,origin, point, angle):
    
        ox, oy = origin
        px, py = point

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        return  qx,qy
    
    
    def pointcloudcallback(self,data):
        self.pc = ros_numpy.numpify(data)
        self.points=np.zeros((self.pc.shape[0],3))
        self.points[:,0]=self.pc['x']
        self.points[:,1]=self.pc['y']
        self.points[:,2]=self.pc['z']
    
        
    def current_velocitycallback(self,msg):
        
        self.lin_x=msg.twist.linear.x
        self.lin_y=msg.twist.linear.y
        self.ang_z=msg.twist.angular.z

        self.x=(math.cos(self.ang_z)*self.lin_x)
        self.y=(math.sin(self.ang_z)*self.lin_x)
        self.vector=[[0,self.x],[0,self.y]]
        self.vel_slope=(self.y/self.x)

        coord=[front_left_distance_from_Lidar,rear_left_distance_from_Lidar,rear_right_distance_from_Lidar,front_right_distance_from_Lidar]
        tr = np.array([[np.cos(self.vel_slope), np.sin(self.vel_slope)], [-np.sin(self.vel_slope), np.cos(self.vel_slope)]])
        mat=np.dot(coord,tr)

        car=Polygon(mat)
        self.xc,self.yc=car.exterior.xy

        slope_c,intercept_c,_,_,_=stats.linregress((mat[0,0],mat[3,0]),(mat[0,1],mat[3,1]))
        
        if not self.points is None:
            z_mask1=self.points[:,2]>-1.6
            z_mask2=self.points[:,2]<-0.5
            z_mask=np.logical_and(z_mask1,z_mask2)
        
            #print(z_mask.shape,self.points.shape)
            self.p_zmasked=self.points[z_mask]
            x_mask1=self.p_zmasked[:,0]>-1.5
            x_mask2=self.p_zmasked[:,0]<25
            self.x_mask=np.logical_and(x_mask1,x_mask2)
            
            
            self.p_xzmasked=self.p_zmasked[self.x_mask]

            scaler=StandardScaler()
            if self.p_xzmasked.size>0:
                X_scaled=scaler.fit_transform(self.p_xzmasked[:,0:2])
                dbscan=DBSCAN(eps=0.48,min_samples=5).fit(self.p_xzmasked[:,0:2])
                self.clusters=dbscan.fit_predict(X_scaled)
                self.cl=self.clusters
                self.clusters=np.array(self.clusters).reshape(-1,1)

            
                self.merge=np.concatenate((self.p_xzmasked,self.clusters),axis=1)

                self.points_2d=self.merge[:,0:2]
                
                hull_list=[]
                for i in range(self.cl.max()+1):
                    a=np.where(self.merge[:,3]==i)
                    hull_list.append(self.points_2d[a])

                self.hull=[]
                for j in range (self.cl.max()+1):
                    if len(hull_list[j])>=3:
                        self.hull.append(ConvexHull(hull_list[j]))

                hull_points=[]
                for e in range(len(self.hull)):
                    min=(self.hull[e].points[self.hull[e].vertices])
                    hull_points.append(min)

                
                hull_points=np.array(hull_points)

                line_list=[]
                for er in range(len(hull_points)):
                    for re in range(len(hull_points[er])-1):
                        ln=LineString([(hull_points[er][re]),(hull_points[er][re+1])])
                        line_list.append(ln)
                

                vertices=[]
                for f in range (len (self.hull)):
                    b=self.hull[f].points[self.hull[f].vertices]
                    vertices.append(b)

                self.vertices=np.array(vertices)
                self.vertices_hulls=np.concatenate(self.vertices)

                self.tri=Delaunay(self.vertices_hulls)
                vor=Voronoi(self.vertices_hulls)
            

                mask_delauney=self.tri.find_simplex(vor.vertices)!=-1          
                self.vertices_voronoi=vor.vertices[mask_delauney]   
                
                
                index_del=np.asarray(np.where(mask_delauney==False)).reshape(-1)

                torles1_vertices=[]
                for iji in range(len(index_del)):
                    list2=np.where(vor.ridge_vertices[:]==index_del[iji])[0]
                    torles1_vertices.append(list2)

                torles1_vertices=np.array(torles1_vertices)
                torles1_vertices=np.unique(np.concatenate(torles1_vertices))
                vor.ridge_vertices=np.delete(vor.ridge_vertices,torles1_vertices,0)
                self.ridges=vor.ridge_vertices
        
                mask1=[]
                for k in vor.ridge_vertices:
                    k=np.asarray(k)
                    c=np.all(k>=0)
                    mask1.append(c)                                 

                self.ridges=self.ridges[mask1]     
                    
                self.vor_vertices_ridges=vor.vertices[self.ridges]        
            

                joining_degree_vertices_id=[]
                for k in range(len(self.vertices_voronoi)):
                    ac=sum(sum(sum(self.vertices_voronoi[k]==self.vor_vertices_ridges)))/2
                    joining_degree_vertices_id.append(ac)
                

                joining_degree_ridges_id=[]
                for aaa in range(self.vor_vertices_ridges.shape[0]):
                    for fff in range(2):
                        bbb=sum(sum(sum(self.vor_vertices_ridges[aaa,fff]==self.vor_vertices_ridges)))/2
                        joining_degree_ridges_id.append(bbb)

                joining_degree_ridges_id=np.array(joining_degree_ridges_id)
                index2=np.where(joining_degree_ridges_id<=1)
                index2=np.array(index2).reshape(-1)
                matrix=np.zeros(((len(index2)),))


                for jjj in range(len(index2)):
                    matrix[jjj]=index2[jjj]//2

                matrix=matrix.astype(int)

                self.vor_vertices_ridges=np.delete(self.vor_vertices_ridges,matrix,0)
                self.ridges=np.delete(self.ridges,matrix,0)
                vertices_index=np.unique(self.ridges)
                self.vertices_voronoi=vor.vertices[vertices_index]


                ransac = linear_model.RANSACRegressor(max_trials=20,min_samples=2)
                if len(self.vertices_voronoi)>3:
                    ransac.fit(self.vertices_voronoi[:,0].reshape(-1,1),self.vertices_voronoi[:,1].reshape(-1,1))
                    
                    self.line_X=np.arange(self.vertices_voronoi[:,0].min(),self.vertices_voronoi[:,0].max())[:, np.newaxis]                    
                    self.line_y_ransac = ransac.predict(self.line_X)

                    self.line_X=np.array([self.line_X[0],self.line_X[-1]])
                    self.line_y_ransac=np.array([self.line_y_ransac[0],self.line_y_ransac[-1]])

                    while self.line_X[-1]<25:
                        self.line_X[-1]=self.line_X[-1]+1


                    if self.line_y_ransac[0]<-5:
                        while self.line_y_ransac[0]<-3:
                            self.line_y_ransac=self.line_y_ransac+1
                    elif self.line_y_ransac[0]>5:
                        while self.line_y_ransac[0]>-3:
                            self.line_y_ransac=self.line_y_ransac-1
                    
                            
                    ransac_line=LineString([(self.line_X[0],self.line_y_ransac[0]),(self.line_X[-1],self.line_y_ransac[-1])]) 

                    intersect_list=[]
                    for intersect in range(len(line_list)):
                        inter=ransac_line.intersects(line_list[intersect])
                        intersect_list.append(inter)
                    intersect_list=np.array(intersect_list)

                    self.slope_ransac_degree=np.rad2deg(math.atan2((self.line_y_ransac[-1]-self.line_y_ransac[0]),(self.line_X[-1]-self.line_X[0])))
                    slope_ransac=((self.line_y_ransac[-1])-self.line_y_ransac[0])/((self.line_X[-1])-self.line_X[0])
                    intercept_ransac=(self.line_y_ransac[0])-(slope_ransac*self.line_X[0])
                    
                    if np.any(intersect_list)==True:
                        try:
                            while np.any(intersect_list)!=False:
                                    if slope_ransac>0:
                                        x,y=self.rotate((self.line_X[0],self.line_y_ransac[0]),(self.line_X[-1],self.line_y_ransac[-1]),np.deg2rad(359))
                                    elif slope_ransac<0:
                                        x,y=self.rotate((self.line_X[0],self.line_y_ransac[0]),(self.line_X[-1],self.line_y_ransac[-1]),np.deg2rad(1))    
                                    self.line_y_ransac[-1]=y
                                    if x<25:
                                        while x>25:
                                            x=x+1
                                        self.line_X[-1]=x
                                    ransac_line=LineString([(self.line_X[0],self.line_y_ransac[0]),(self.line_X[-1],self.line_y_ransac[-1])])
                                    intersect_list=[]
                                    for intersect in range(len(line_list)):
                                        inter=ransac_line.intersects(line_list[intersect])
                                        intersect_list.append(inter)
                                    intersect_list=np.array(intersect_list)
                        except:
                            pass
                        
                    
                    slope_ransac=((self.line_y_ransac[-1])-self.line_y_ransac[0])/((self.line_X[-1])-self.line_X[0])
                    intercept_ransac=(self.line_y_ransac[0])-(slope_ransac*self.line_X[0])
                    
                    y_ransac_vertices=(slope_ransac*self.vertices_hulls[:,0])+intercept_ransac
                    
                    y_vertices_left_mask=self.vertices_hulls[:,1]>y_ransac_vertices
                    y_vertices_right_mask=self.vertices_hulls[:,1]<y_ransac_vertices

                    left_side=self.vertices_hulls[y_vertices_left_mask]
                    right_side=self.vertices_hulls[y_vertices_right_mask]
                    
                    left_side=left_side[left_side[:,0].argsort()]
                    right_side=right_side[right_side[:,0].argsort()]

                    lower_left= []
                    
                    for pp in reversed(left_side):
                        while len(lower_left) >= 2 and self.cross(lower_left[-2], lower_left[-1], pp)\
                                <= 0:
                            lower_left.pop()
                        lower_left.append(pp)

                    upper_left = []

                    for pp in left_side:
                        while len(upper_left) >= 2 and self.cross(upper_left[-2], upper_left[-1], pp) \
                                <= 0:
                            upper_left.pop()
                        upper_left.append(pp)

                    self.upper_left=np.array(upper_left)
                    self.lower_left=np.array(lower_left)

                    lower_right= []
                    
                    for pe in right_side:
                        while len(lower_right) >= 2 and self.cross(lower_right[-2], lower_right[-1], pe)\
                                <= 0:
                            lower_right.pop()
                        lower_right.append(pe)

                    upper_right = []

                    for pe in reversed(right_side):
                        while len(upper_right) >= 2 and self.cross(upper_right[-2], upper_right[-1], pe) \
                                <= 0:
                            upper_right.pop()
                        upper_right.append(pe)

                    
                    self.upper_right=np.array(upper_right)
                    self.lower_right=np.array(lower_right)

                    left_polygons=np.concatenate((self.upper_left,self.lower_left))
                    right_polygons=np.concatenate((self.upper_right,self.lower_right))

                    if len(left_polygons)>2 and len(right_polygons)>2:
                        polygon_left=Polygon(left_polygons)
                        polygon_right=Polygon(right_polygons)
                        self.xl,self.yl=polygon_left.exterior.xy
                        self.xr,self.yr=polygon_right.exterior.xy

                        mask_vor_L, mask_vor_R=[],[]
                        for ib in range(len(self.vertices_voronoi)):
                            vp=Point(self.vertices_voronoi[ib,0],self.vertices_voronoi[ib,1])
                            lp=polygon_left.contains(vp)
                            rp=polygon_right.contains(vp)
                            mask_vor_L.append(lp)
                            mask_vor_R.append(rp)
                        mask_vor_L=np.array(mask_vor_L)
                        mask_vor_R=np.array(mask_vor_R)
                        mask_vertices=np.logical_or(mask_vor_L,mask_vor_R)    
                        
                        self.vertices_voronoi=self.vertices_voronoi[~mask_vertices]

                        torles_ridges=[]
                        for ha in vertices_index[np.where(mask_vertices==True)]:
                            ab=np.where(self.ridges==ha)[0]
                            torles_ridges.append(ab)

                        torles_ridges=np.array(torles_ridges)
                        if len(torles_ridges)>0:
                            torles_ridges=np.unique(np.concatenate(torles_ridges))
                            self.ridges=np.delete(self.ridges,torles_ridges,0)

                            self.vor_vertices_ridges=vor.vertices[self.ridges]

                        x_car=(-intercept_c+self.vertices_voronoi[:,1])/slope_c 
                        x_car_mask=self.vertices_voronoi[:,0]>x_car

                        self.vertices_voronoi=self.vertices_voronoi[x_car_mask]
                        """
                        self.vertices_XY=[]
                        
                        for g in range(len(self.vertices_voronoi)):
                            f=Point([self.vertices_voronoi[g,0],self.vertices_voronoi[g,1]]).xy
                            self.vertices_XY.append(f)

                        self.vertices_XY=np.array(self.vertices_XY)
                        self.vertices_XY=self.vertices_XY.reshape(self.vertices_XY.shape[0],self.vertices_XY.shape[1])

                        zeros=np.zeros(len(self.vertices_XY))

                        self.vertices_XY=np.column_stack((self.vertices_XY,zeros))

                        #print(self.vertices_voronoi.shape)
                        
                    
                        
                        torles_ridges2=[]
                        
                        for ha2 in vertices_index[np.where(x_car_mask==False)]:
                            ab2=np.where(ridges==ha2)[0]
                            torles_ridges2.append(ab2)

                        torles_ridges2=np.array(torles_ridges2)
                        if len(torles_ridges2)>0:
                            torles_ridges2=np.unique(np.concatenate(torles_ridges2))
                            ridges=np.delete(ridges,torles_ridges2,0)

                            self.vor_vertices_ridges=vor.vertices[ridges]
                        
                        
                        fig = plt.figure()
                        #plt.triplot(self.vertices_hulls[:,0], self.vertices_hulls[:,1], self.tri.simplices.copy())       
                        plt.scatter(self.vertices_voronoi[:,0],self.vertices_voronoi[:,1],c='g')                          
                        #plt.scatter(zero[0],zero[1],c='orange')                                                          
                        plt.scatter(self.p_xzmasked[:,0],self.p_xzmasked[:,1],c='orange')                                                  
                        #plt.scatter(self.vertices_hulls[:,0],self.vertices_hulls[:,1],c="m")                             
                        #plt.scatter(self.vertices_voronoi[:,0],np.polyval(self.p_voronoi_vertices,self.vertices_voronoi[:,0]),c='r')  
                        #plt.plot(self.upper[:,0],self.upper[:,1],c="b")
                        #plt.plot(self.lower[:,0],self.lower[:,1],c='r')
                        #plt.plot(self.upper_right[:,0],np.polyval(self.p_upper_R,self.upper_right[:,0]),c='g')
                        
                        for e in range(len(self.hull)):
                            #plt.scatter(self.p_xzmasked[:,0],self.p_xzmasked[:,1],c=self.cl)                                                  #klaszterezett eredeti pontok
                            plt.plot(self.hull[e].points[self.hull[e].vertices,0], self.hull[e].points[self.hull[e].vertices,1], 'k--', lw=2)  
                        

                        for index in range(len(self.vor_vertices_ridges)):
                            plt.plot(self.vor_vertices_ridges[index,:,0],self.vor_vertices_ridges[index,:,1],c='k')                  #Voronoi kpontok/ref jel
                        
                        
                    
                        #plt.scatter(self.lower_left[:,0],self.lower_left[:,1],c='b')
                        #plt.scatter(self.upper_left[:,0],self.upper_left[:,1],c='m')
                        
                    
                        #plt.plot(self.vector[0],self.vector[1],c='b')
                        plt.plot(self.line_X,self.line_y_ransac,c='c')
                        
                        
                        
                        
                        #plt.plot(self.vertices_voronoi[:,0],y_upper_L,c='m')
                        #plt.plot(left_side[:,0],left_side[:,1])
                        #plt.plot(right_side[:,0],right_side[:,1])
                        #plt.scatter(self.upper_right[:,0],self.upper_right[:,1],c='b')
                        
                        #plt.plot(self.lower_right[:,0],self.lower_right[:,1],c='r')
                        #plt.plot(self.upper_right[:,0],self.upper_right[:,1],c='b')
                        #plt.plot(self.lower_left[:,0],self.lower_left[:,1],c='c')
                        #plt.plot(self.upper_left[:,0],self.upper_left[:,1],c='m')
                        #plt.scatter(self.upper_left[:,0],self.upper_left[:,1],c='m')
                        #plt.plot(self.vertices_voronoi[:,0],y_upper_R,c='c')
                        #plt.plot(self.vertices_voronoi[:,0],y_upper_L,c='orange')
                        #plt.scatter(self.vertices_voronoi[:,0],self.vertices_voronoi[:,1],c='g')
                        #plt.plot(self.vertices_right_upper[:,0],self.y_upper_R,c='y')
                        plt.plot(self.xl,self.yl,c="r")
                        plt.plot(self.xr,self.yr,c="b")
                        plt.plot(self.xc,self.yc,c='k')
                        plt.plot(self.vector[0],self.vector[1],c='m')
                        #plt.plot(x_car,self.vertices_voronoi,c='y')
                                
                        plt.axis('equal')
                        plt.show()
                        """


def callback():
    Ts=TopicSubscriber()    

    rospy.init_node("challange1", anonymous=True)
    
    velodyne_sub=rospy.Subscriber("/velodyne_points", senmsg.PointCloud2,Ts.pointcloudcallback)
    velocity_sub=rospy.Subscriber("/current_velocity", geomsg.TwistStamped,Ts.current_velocitycallback)
    #rospy.spin()
    
    topic = 'Reference_line'
    publisher = rospy.Publisher(topic,MarkerArray,queue_size=2)
    rate=rospy.Rate(10)

    count=0
    MARKERS_MAX=20

    markerArray= MarkerArray()
    marker = Marker()
    marker.header.frame_id = "/velodyne"
    marker.type = marker.SPHERE_LIST
    marker.action = marker.ADD
    marker.scale.x = 1.5
    marker.scale.y = 1.5
    marker.scale.z = 1.5
    marker.color.a = 124.0
    marker.color.r = 0.0
    marker.color.b = 0.0
    marker.color.g = 252.0
    

    while not rospy.is_shutdown():
        marker.lifetime=rospy.Duration(0)
        
        if not Ts.vertices_voronoi is None:
            

            marker.points=[]
            
            for g in range(len(Ts.vertices_voronoi)):
                p=geomsg.Point
                p.x=Ts.vertices_voronoi[g,0]
                p.y=Ts.vertices_voronoi[g,1]
                p.z=0

            marker.points.append(p)
            #print(list(marker.points))
            if(count > MARKERS_MAX):
                markerArray.markers.pop(0)

            markerArray.markers.append(marker)
            
            id = 0
            for m in markerArray.markers:
                m.id = id
                id += 1

            #print(markerArray)
                    
            publisher.publish(markerArray)
            
            

            rate.sleep()
        #"""
if __name__ == '__main__':
    try:
        callback()
        
    except rospy.ROSInterruptException:
        pass


    