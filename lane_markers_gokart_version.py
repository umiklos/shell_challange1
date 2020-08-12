#!/usr/bin/env python

import rospy
import numpy as np
import sensor_msgs.msg as senmsg
from sensor_msgs.msg import PointCloud2
import ros_numpy
import shapely.geometry as sg
import shapely
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from scipy.spatial import Voronoi
import visualization_msgs.msg as vismsg
import geometry_msgs.msg as geomsg
from shapely.ops import shared_paths

reference_line=None
polygon=None
polygon_with_origo=None


class reference():
    def pointcloudcallback(self,data):
        global polygon,reference_line,polygon_with_origo,second_half_interval
        pc = ros_numpy.numpify(data)
        points=np.zeros((pc.shape[0],3))
        points[:,0]=pc['x']
        points[:,1]=pc['y']
        points[:,2]=pc['z']
        
        origo=sg.Point(0,0)
        origo_array=np.asarray(origo).reshape(-1,2)

        if (len(points))>1: 
            clusters, n_clusters=self.get_euclidean_clustering(points,0.45,10)
            
            
            if n_clusters>1:
                segmented_points=np.column_stack((points[:,0:2],clusters))
                index,count=np.unique(clusters,return_counts=True)
                selected_column=np.where(segmented_points[:,2]==index[np.argmax(count)])
                selected_points=segmented_points[selected_column]
                
            elif n_clusters==1 and -1 in set(clusters):
                segmented_points=np.column_stack((points[:,0:2],clusters))
                selected_column=np.where(segmented_points[:,2]==0)
                selected_points=segmented_points[selected_column]
            elif n_clusters==0:
                selected_points=[]                
            else:
                selected_points=points
                        

            if len(selected_points)>2:

                polygon, rotated_rect=self.make_polygon(selected_points[:,0:2])

                polygon_points_with_origo=np.concatenate((origo_array,selected_points[:,0:2]))
                polygon_with_origo,rotated_rect_with_origo=self.make_polygon(polygon_points_with_origo[:,0:2])

                if type(polygon) is sg.polygon.Polygon and type(polygon_with_origo) is sg.polygon.Polygon :
                    xy=np.column_stack((polygon.exterior.coords.xy[0],polygon.exterior.coords.xy[1]))
                    line=[]
                    for i in range(1,len(xy)):
                        line.append(sg.LineString([(xy[i-1,0],xy[i-1,1]),(xy[i,0],xy[i,1])]))
                    pp0,pp1=self.get_p0(rotated_rect_with_origo,origo)
                    p0,p1=self.get_p0(rotated_rect,origo)

                    
                    p0c=p0.centroid
                    p1c=p1.centroid

                    midline=sg.LineString([p0c,p1c])
                    polygon_first=self.get_polygon_first_side(midline,line,p0)

                    min_distance=p0.hausdorff_distance(polygon_first)
                    max_distance=p0.distance(p1)
                    p0_pp0_distance=pp0.distance(p0)

                    
                    first_half=self.cross(pp0,polygon_with_origo,0.2,p0_pp0_distance)
                    second_half=self.cross(p0,polygon,min_distance,max_distance)                    
                    
                    if len(first_half)>0 and len(second_half)>0:  
                        merged_points=np.concatenate((first_half,second_half))
                        
                        if len(merged_points)>3:
                            vor=Voronoi(merged_points)
                            vertices=self.get_vertices(vor,polygon_with_origo)
                            if len(vertices)>2:
                                mymodell=np.poly1d(np.polyfit(vertices[:,0],vertices[:,1],3))
                                myline=np.linspace(vertices[:,0].min(),vertices[:,0].max(),len(vertices))

                                reference_line=np.column_stack((myline,mymodell(myline)))


            else:
                rospy.logwarn('no polygon can be formed')
                                


            
    def make_polygon(self,points):
        points=sg.MultiPoint(points)
        polygon=points.convex_hull
        min_rotated_rect=points.minimum_rotated_rectangle
        return polygon,min_rotated_rect    

    def get_euclidean_clustering(self,points,EPS,MIN_SAMPLES):
        scaler=StandardScaler()
        X_scaled=scaler.fit_transform(points[:,0:2])        
        dbscan=DBSCAN(eps=EPS,min_samples=MIN_SAMPLES).fit(points[:,0:2])
        clusters=dbscan.fit_predict(X_scaled)
        labels=set(clusters)

        if -1 in labels:
            n_clusters=len(labels)-1
        else:
            n_clusters=len(labels)
        
        return clusters, n_clusters
    
    def get_p0(self,rotated_rect,origo):
        
        p0,p1,p2,p3=self.get_p_coordinates(rotated_rect)
        dist0,dist1,dist2,dist3=self.get_distances(p0,p1,p2,p3,origo)
        

        if dist0<dist1 and dist0<dist2 and dist0<dist3 :
            return p0,p1
        elif dist1<dist0 and dist1<dist2 and dist1<dist3 :
            return p1,p0
        elif dist2<dist0 and dist2<dist1 and dist2<dist3 :
            return p2,p3
        else:
            return p3,p2
    
    def get_p_coordinates(self,rotated_rect):
        xr=rotated_rect.exterior.coords.xy[0]
        yr=rotated_rect.exterior.coords.xy[1]

        p0=sg.LineString([(xr[3],yr[3]),(xr[4],yr[4])])
        p1=sg.LineString([(xr[1],yr[1]),(xr[2],yr[2])])
        p2=sg.LineString([(xr[0],yr[0]),(xr[1],yr[1])])
        p3=sg.LineString([(xr[2],yr[2]),(xr[3],yr[3])])
        return p0,p1,p2,p3

    def get_distances(self,p0,p1,p2,p3,origo):
        dist0=origo.distance(p0.centroid)
        dist1=origo.distance(p1.centroid)
        dist2=origo.distance(p2.centroid)
        dist3=origo.distance(p3.centroid)
        return dist0,dist1,dist2,dist3



    def cross(self,p0,polygon,min_distance,max_distance):
        
        iterator=self.get_iterator(min_distance,max_distance)
        crosslines=[]
        points=[]
        if len(iterator)>2:
            for i in range(len(iterator)):
                crosslines.append(p0.parallel_offset(iterator[i],'left').intersection(polygon))
                
            for i in range(len(crosslines)):
                if  type (crosslines[i]) is sg.linestring.LineString:
                    for j in range(2):
                        x1=crosslines[i].xy[0][j]
                        y1=crosslines[i].xy[1][j]
                        points.append([x1,y1])
        points=np.array(points)
        return points

    def get_vertices(self,vor,free_space_sim):
        mask=[]
        vertices=vor.vertices
        
        for i in range(len(vor.vertices)):
            
            mask.append(free_space_sim.contains(sg.Point(vor.vertices[i,0],vor.vertices[i,1])))
            
        vertices=vertices[mask]
        return vertices

    def get_polygon_first_side(self,midline,line,p0):
        index=[]
        
        for i in range(len(line)):
            if midline.crosses(line[i])==True:
                index.append(i)

        if len(index)<2:
            return p0
        else:
            p1=line[index[0]]
            p2=line[index[1]]

            if p1.distance(p0) < p2.distance(p0):
                return p1
            else:
                return p2
        


    def get_iterator(self,min_distance,max_distance):
        distance=max_distance-min_distance
        if distance<1.5:
            iterator=np.arange(min_distance+0.1,max_distance-0.1,0.2)
        elif distance<3 and distance>1.5:
            iterator=np.arange(min_distance+0.2,max_distance-0.2,0.5)
        elif distance>3 and distance<5:
            iterator=np.arange(min_distance+0.4,max_distance-0.4,0.8)
        elif distance>5:
            iterator=np.arange(min_distance+0.4,max_distance-0.4,1.2)
        return iterator
        
    
def listener():
    r=reference()
    rospy.init_node('listener', anonymous=True)
    
    rospy.Subscriber("/lidar_lane",senmsg.PointCloud2, r.pointcloudcallback)

    #"""
    pub_free = rospy.Publisher("free_space_polygon", vismsg.Marker, queue_size=1)
    pub_ref=rospy.Publisher("reference_line",vismsg.Marker,queue_size=1)
    
    rate=rospy.Rate(10)

    mark_f = vismsg.Marker()
    mark_f.header.frame_id = "/right_os1/os1_lidar"
    mark_f.type = mark_f.LINE_STRIP
    mark_f.action = mark_f.ADD
    mark_f.scale.x = 0.2
    mark_f.color.r = 0.1
    mark_f.color.g = 0.4
    mark_f.color.b = 0.9
    mark_f.color.a = 0.9 # 90% visibility
    mark_f.pose.orientation.x = mark_f.pose.orientation.y = mark_f.pose.orientation.z = 0.0
    mark_f.pose.orientation.w = 1.0
    mark_f.pose.position.x = mark_f.pose.position.y = mark_f.pose.position.z = 0.0

    mark_r = vismsg.Marker()
    
    mark_r.header.frame_id = "/right_os1/os1_lidar"
    mark_r.type = mark_r.LINE_STRIP
    mark_r.action = mark_r.ADD
    mark_r.scale.x = 0.2
    mark_r.color.r = 0.1
    mark_r.color.g = 0.9
    mark_r.color.b = 0.6
    mark_r.color.a = 0.9 # 90% visibility
    mark_r.pose.orientation.x = mark_r.pose.orientation.y = mark_r.pose.orientation.z = 0.0
    mark_r.pose.orientation.w = 1.0
    mark_r.pose.position.x = mark_r.pose.position.y = mark_r.pose.position.z = 0.0

   
    
    while not rospy.is_shutdown():
        mark_r.header.stamp=rospy.Time.now()
        mark_f.header.stamp=rospy.Time.now()
        if polygon is not None and polygon_with_origo is not None :

        
            # marker line points
            mark_f.points=[]
            mark_r.points=[]
            if type(polygon) is sg.polygon.Polygon:
                for l in polygon.exterior.coords[1:]: # the last point is the same as the first
                    p = geomsg.Point(); p.x = l[0]; p.y = l[1]; p.z = -0.3
                    mark_f.points.append(p)
            if  reference_line is not None:
                for j in reference_line:
                    p=geomsg.Point(); p.x=j[0]; p.y=j[1]; p.z=-0.3
                    mark_r.points.append(p)
            else:
                rospy.logwarn("no reference line can be created")
               
                

            pub_ref.publish(mark_r)
            pub_free.publish(mark_f)
    rate.sleep()
    #"""       
            
    

    #rospy.spin()

if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        pass