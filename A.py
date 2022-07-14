from ast import Global
from cProfile import label
from cmath import cos
import collections
from operator import iadd
from pickle import TRUE
import re
import sys

import bpy
import math
import random
from mathutils import Vector
from gpu_extras.batch import batch_for_shader
import shapely.geometry

import platform
import time
import os
import numpy as np

# import thread

C = bpy.context
D = bpy.data
room_len = 20
room_wid = 20



def check_if_cross_wall(object):
    """
    If cross the wall return true, else retuen false

    verts: (Vector((x, y, z)), Vector((a, b, c)))
    """

    verts = [v.co for v in object.data.vertices]
    #obb_local = ret_obb(verts)
    mat = object.matrix_world
    point_world = [mat @ v for v in verts]
    temp_list = []
    for x in range(len(point_world)):
        # 1 means this point cross the wall, 0 not
        if point_world[x][:][0] < room_len and point_world[x][:][0] > 0 and point_world[x][:][1] < room_wid and point_world[x][:][1] > 0:
            temp_list.append(0)
        else: 
            temp_list.append(1)
    if 1 in temp_list:
        #print(object.name, "cross wall")
        return True
    else:
        #print(object.name, "in room")
        return False

def point_list(ob_list):
    """
    Description:

    bpy's object property format is like: [<bpy id property array [2]>, ...,  <bpy id property array [2]>]
    we need [(x_1, y_1), ..., (x_n, y_n)]

    Return: [(x_1, y_1), ..., (x_n, y_n)]
    """
    list = []
    for iter in ob_list:
        x = iter[:]
        list.append(x)

    return list

def orient_vector(object):
    """
    Return: (x, y, z)
    """
    # center = object.location
    orient_vec = object.matrix_world @ (Vector((1, 0, 0, 0)))

    if bpy.context.object.name == 'wall_1':
        orient_vec = object.matrix_world @ (Vector((0, 1, 0, 0)))
    elif bpy.context.object.name == 'wall_2':
        orient_vec = object.matrix_world @ (Vector((1, 0, 0, 0)))
    elif bpy.context.object.name == 'wall_3':
        orient_vec = object.matrix_world @ (Vector((0, -1, 0, 0)))
    elif bpy.context.object.name == 'wall_3':
        orient_vec = object.matrix_world @ (Vector((-1, 0, 0, 0)))
    else:
        orient_vec = object.matrix_world @ (Vector((1, 0, 0, 0)))
    
    return orient_vec[0:3]

def activate(objName):
    bpy.context.view_layer.objects.active=bpy.data.objects[objName]

def ret_obb(verts):
    """
    Description:
    Get verts of object and calculate object's OBB

    Return:

    """
    points = np.asarray(verts)
    means = np.mean(points, axis=1)

    cov = np.cov(points, y = None,rowvar = 0,bias = 1)

    v, vect = np.linalg.eig(cov)

    tvect = np.transpose(vect)
    points_r = np.dot(points, np.linalg.inv(tvect))

    co_min = np.min(points_r, axis=0)
    co_max = np.max(points_r, axis=0)

    xmin, xmax = co_min[0], co_max[0]
    ymin, ymax = co_min[1], co_max[1]
    zmin, zmax = co_min[2], co_max[2]

    xdif = (xmax - xmin) * 0.5
    ydif = (ymax - ymin) * 0.5
    zdif = (zmax - zmin) * 0.5

    cx = xmin + xdif
    cy = ymin + ydif
    cz = zmin + zdif

    corners = np.array([
        [cx - xdif, cy - ydif, cz - zdif],
        [cx - xdif, cy + ydif, cz - zdif],
        [cx + xdif, cy + ydif, cz - zdif],
        [cx + xdif, cy - ydif, cz - zdif],
        [cx - xdif, cy - ydif, cz + zdif],
        [cx - xdif, cy + ydif, cz + zdif],
        [cx + xdif, cy + ydif, cz + zdif],
        [cx + xdif, cy - ydif, cz + zdif],
    ])

    corners = np.dot(corners, tvect)

    return [Vector((el[0], el[1], el[2])) for el in corners]


def Align():
    '''
    Description:
    choose all objects in nected collection then aline to one direction

    '''
    
    def ViewRectangle(obj):
            """
            Description:
            Define object's visible area, after align()

            """
            if Viewable(obj):
                temp = shapely.geometry.MultiPoint(obj["bottom_poly"]).convex_hull
                # x_i = (a, b)

                x_0 = temp.exterior.coords[0]
                x_1 = temp.exterior.coords[1]
                x_2 = temp.exterior.coords[2]
                x_3 = temp.exterior.coords[3]
                
                obj.select_get()
                point_0 = x_3
                point_1 = x_2
                point_2 = (x_2[0] + 3, x_2[1])
                point_3 = (x_3[0] + 3, x_3[1])

                bpy.ops.mesh.primitive_plane_add()
                bpy.context.object.scale[0] = 3
                bpy.context.object.scale[1] = x_1[1] - x_0[1]
                bpy.context.object.location[0] = (x_0[0] + (x_2[0] - x_1[0])/2) + (x_2[0] - x_1[0])/2 + 1.5
                bpy.context.object.location[1] = x_0[1] + (x_1[1] - x_0[1])/2
                bpy.context.object.name = obj.name + '_' + 'viewRec'

  
                bpy.context.object.select_set(True)
                activate(obj.name)
                bpy.ops.object.join()


    for object in bpy.data.collections["furniture"].all_objects:
        object.select_set(True)
        
        bpy.ops.transform.transform(mode='ALIGN', value=(0, 0, 0, 0), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)
        object.location.z = 0
        object.select_set(False)
        
    for object in bpy.data.collections["furniture"].all_objects:
        name = object.name
        bpy.data.objects[name].select_set(True)
        accessableArea(object)       

# 
        ViewRectangle(object)



def Viewable(object):
    """
    Description:
    Check if object is visible object.

    Return: True or False
    """
    
    object_front_name = object.name.split("_")[0]
    object_last_name = object.name.split("_")[1]
    
    if object_last_name == 'TV' or object_front_name == 'cupboard':
        return True
    else:
        return False


def Visible_cost(group):
    """
    Description: If some object in one object's visible area, then cost will rise. 
    Here we define cost as the area of intersection of one object with another object's visible area
    """
    # group: bpy.data.collections["x"]
    for obj in group.all_objects:
        if len(obj.name.split("_")) < 3:
            if Viewable(obj):
                # name = obj.name + '_' + 'viewRec'

                obj_child_name = [obj.children[0].name][0]
                verts = [v.co[:] for v in D.objects[obj_child_name].data.vertices]

                plane_poly = shapely.geometry.MultiPoint(verts).convex_hull
                
                area = 0

                for obj_other in group.all_objects:
                    
                    if obj_other.name != 'table_TV_viewRec' and obj_other.name != obj.name:
                        verts_o = point_list(D.objects[obj_other.name]["bottom_poly"])
                        x = shapely.geometry.MultiPoint(verts_o).convex_hull

                        if x.intersects(plane_poly):
                            area += x.intersection(plane_poly).area
            else:
                area = 0
    return area
    

def touchable(object):
    """
    Description:
    define if an object touchable

    Return:True or False
    """
    if object.name.split("_")[0] == 'sofa' or 'table' or 'chair':
        return True
    
    elif object.name.split("_")[0] == 'potting' or 'water_cooler' or 'cupboard':
        return False

def accessableArea(object):
    '''
    Description:
    One property for object.

    Return: object's acessable area(4 points)

    '''
    if object.name != 'door' or 'Plane' or 'window':
        # object.select_set(True)
        bpy.ops.object.mode_set(mode = 'OBJECT')
        verts = [v.co for v in object.data.vertices]
        obb_local = ret_obb(verts)
        mat = object.matrix_world
        obb_world = [mat @ v for v in obb_local]

        # Visualization
        # bpy.ops.object.mode_set(mode = 'EDIT')

        # # draw with GPU Module
        # coords = [(v[0], v[1], v[2]) for v in obb_world]

        # indices = (
        #     (0, 1), (1, 2), (2, 3), (3, 0),
        #     (4, 5), (5, 6), (6, 7), (7, 4),
        #     (0, 4), (1, 5), (2, 6), (3, 7))

        # shader = gpu.shader.from_builtin('3D_UNIFORM_COLOR')
        # batch = batch_for_shader(shader, 'LINES', {"pos": coords}, indices=indices)


        # def draw():
        #     shader.bind()
        #     shader.uniform_float("color", (1, 0, 0, 1))
        #     batch.draw(shader)

        # bpy.types.SpaceView3D.draw_handler_add(draw, (), 'WINDOW', 'POST_VIEW')

        # EXAMPLE of object's bottom_poly: [(x_1, x_2), (x_3, x_4), ... , (x_k, x_k+1)]
        list = []
        for x in range(len(obb_world)):
            if abs(obb_world[x][:][2]) <= 1:
                list.append(obb_world[x][:][0:2])
        
        object["bottom_poly"] = list
        object["accessArea"] = shapely.affinity.scale(shapely.geometry.MultiPoint(object["bottom_poly"]).convex_hull, xfact = 1.2, yfact = 1.2).exterior.coords[:]

    else:
        pass

    # box = object.bound_box
    # p = [object.matrix_world @ Vector(corner) for corner in box]




    

def CheckIfHit(object_1, object_2):
    """
    Description:
    check if object_1's bottom hit to object_2's bottom

    Return:
    True or False
    """
    bottom_point_1 = point_list(object_1["bottom_poly"])
    bottom_point_2 = point_list(object_2["bottom_poly"])

    obj_1_bottom_poly = shapely.geometry.MultiPoint(bottom_point_1).convex_hull
    obj_2_bottom_poly = shapely.geometry.MultiPoint(bottom_point_2).convex_hull

    return obj_1_bottom_poly.intersects(obj_2_bottom_poly)

def CheckIntersect(object_1, object_2):
    """
    Description:
    check if object_1's bottom in object_2's accessible area and return intersected point list

    Return:
    list of object_1's bottom points which intersected with object_2's access area
    """

    temp = point_list(object_2["accessArea"])
    obj_2_access_poly = shapely.geometry.MultiPoint(temp).convex_hull

    intersectPoint = []

    bottom_list = point_list(object_1["bottom_poly"])

    for i in range(len(bottom_list)):
        i = shapely.geometry.Point(bottom_list[i])
        if obj_2_access_poly.intersects(i):
            intersectPoint.append(i)

    if not bottom_list:
        return False
    else:
        return intersectPoint
    
def select_from_collection(some_collection):
    """ Recursively select objects from the specified collection """

    list = []
    for a_collection in some_collection.children:
        select_from_collection(a_collection)
    for obj in some_collection.objects:
        obj.select_set(True)
        list.append(obj)
        obj.select_set(False)

    return list

def calc1(access_point_list, object_2):
    """
    Description:
    Calculate the cost when A in B's access area but not hit with B

    list: list of points in access area

    object_2: object where A's points in its access area

    Return: 
    cost(double)
    """

    temp = object_2["bottom_poly"]
    # obj_2_poly_context = {'type': 'MULTIPOLYGON',
    # 'coordinates': [[[list(temp[0]), list(temp[1]), list(temp[2]), list(temp[3])]]]}

    # # to calculate the distance from point to object_2's edge, first we set up object_2's shape
    # obj_2_poly_shape = shapely.geometry.asShape(obj_2_poly_context)
    obj_2_bottom_poly = shapely.geometry.MultiPoint(temp).convex_hull

    center = obj_2_bottom_poly.centroid

    sum_cost = 0
    for i in range(len(access_point_list)):
        point = shapely.geometry.Point(access_point_list[i])
        dis = center.distance(point)
        cost = math.exp(1/dis)
        sum_cost += cost

    return sum_cost

def calc2(list_, object_1, object_2):
    """
    Description:
    Calculate the cost when A in B's access area and hit with B
    use intersect area + calc1(~) as cost

    """

    point_cost = calc1(list_, object_2)

    bottom_point_1 = point_list(object_1["bottom_poly"])
    bottom_point_2 = point_list(object_2["bottom_poly"])
    obj_1_bottom_poly = shapely.geometry.MultiPoint(bottom_point_1).convex_hull
    obj_2_bottom_poly = shapely.geometry.MultiPoint(bottom_point_2).convex_hull

    inter_area = obj_1_bottom_poly.intersection(obj_2_bottom_poly).area

    
    return point_cost + inter_area
    
def nearest_wall(object, wall_group):
    """
    Description:
    find the nearest wall to an object.

    """
    name = object.name
    obj = bpy.data.objects[name]

    closet_wall = None
    distance = -1

    def get_distance(p1, p2) :
        [x1, y1, z1] = p1
        [x2, y2, z2] = p2
        return (((x2-x1)**2) + ((y2-y1)**2) + ((z2-z1)**2)) ** (1/2)

    for o in wall_group.objects:
        if obj == o or not o.type == 'MESH':
            continue
        d = get_distance(obj.location, o.location)
        if distance < 0 or d < distance:
            distance = d
            closet_wall = o

    if not closet_wall is None:
        return closet_wall
        # # print("object", closet_wall.name, "is closest to", obj.name, "with distance of", distance)
    else:
        return 0
        # # print("no mesh objects found!")

def access_cost(group):
    """
    Description: if one object was set into an access area of another object, 
    this function calculate the cost for this sence

    Return: cost
    """

    list = select_from_collection(group)
    for i in range(len(list)):
        for j in range(len(list)-i):
            # if they are 2 different objects
            if i != i+j:
                # if A in B's access area:(A count is list[i], B count is list[i+j])
                if CheckIntersect(list[i], list[i+j]) != False:
                    # if A not hit or cover B
                    Intersect_point_list = CheckIntersect(list[i], list[i+j])
                    if CheckIfHit(list[i], list[i+j]) == False:
                        # # print(Intersect_point_list)
                        cost = calc1(Intersect_point_list, list[i+j])
                    # if A hit or cover B
                    else:
                        cost = calc2(Intersect_point_list, list[i], list[i+j])
                else:
                    continue
    
    return cost
        



def orient_cost(group):
    """
    Description:
    while object orient same with wall's, cost will down.

    """
    cost = 0
    wall_collection = D.collections['wall']
    for obj in group.objects:
        nearestWall = nearest_wall(obj, wall_collection)
        obj_vec = orient_vector(obj)
        wall_vec = orient_vector(nearestWall)
        a_1 = obj_vec[0]
        a_2 = obj_vec[1]
        a_3 = obj_vec[2]
        b_1 = wall_vec[0]
        b_2 = wall_vec[1]
        b_3 = wall_vec[2]

        upper = a_1*b_1 + a_2*b_2 + a_3*b_3
        lower = math.sqrt(a_1*a_1 + a_2*a_2 + a_3*a_3) * math.sqrt(b_1*b_1 + b_2*b_2 + b_3*b_3)
        cos_theta = upper/lower 
        temp = cos_theta + 1
        cost += temp

    return cost
        

def attract_cost(group):
    """
    Description:
    To solve simulated annealing process without wall limitation problem, define attraction cost function to avoid 
    objects scattered too much.

    Return: cost
    """

    max_lenth = 10# m, lenth of room
    def func(x):
        if x/max_lenth > 1:
            return math.exp(x/max_lenth - 1)
        else:
            return x/max_lenth

    list = select_from_collection(group)
    sum = 0
    for i in range(len(list)):
        for j in range(len(list)-i):
            if i != i+j:
                x = point_list(list[i]["bottom_poly"])
                y = point_list(list[i+j]["bottom_poly"])
                point_a = shapely.geometry.MultiPoint(x).convex_hull.centroid
                point_b = shapely.geometry.MultiPoint(y).convex_hull.centroid
                dis = point_a.distance(point_b)
                sum += dis
    average_dis = sum/len(list)
    return func(average_dis)


# def orient_cost(group):
#     for object in group.objects:
#         wall = find_nearest(object)
#         wall_vec = orient_vector(wall)
#         object_vec = orient_vector(object)


def total_cost(group):
    """
    Return attract cost, normal cost and visible cost.
    """
    # group: bpy.data.collections["x"]
    a = attract_cost(group)
    b = access_cost(group)
    c = Visible_cost(group)
    d = orient_cost(group)
    #print("*a:" , a)
    #print("**b:" , b)
    #print("***c:" , c)
    #print("****d:" , d)

    return 0.5*a + 10*b + 2*c + 5*d

def move_rotate(obj, T, T_0):
    """
    Description:move and rotate object randomly
    
    """
    if len(obj.name.split("_")) < 3:
        

        std = math.sqrt(T/10)
        m = np.random.normal(0, std, 1)
        if np.random.randint(2):
            if np.random.randint(2):
                obj.location.x += m
            else:
                obj.location.y += m
        else:
            # rootate
            std = math.pi * (T/T_0)
            r = np.random.normal(0, std, 1)
            obj.rotation_euler.z += r
            bpy.context.view_layer.update()

        
    
def simulated_annealing(group):
    """
    Description:
    SA
    """
    #         roomPoints= [geometry.Point(0,0),geometry.Point(100,0)
    #                      ,geometry.Point(100,100),geometry.Point(0,100)]

    # num: iteration times
    # alpha: cooling index
    # group: bpy.context.collection
    T_0 = 100
    T_F = 10
    num = 150
    alpha = 0.9
    T = T_0
    global beta    
    # initial arrangement
    c_0 = total_cost(group)


    T = T_0
    while T > T_F:
        for i in range(num):
            count = 0
            bpy.ops.ed.undo_push()

            for obj in group.objects:
                #print("*", obj.name)
                if len(obj.name.split("_")) < 3:
                    activate(obj.name)
                else:
                    continue
                obj.select_set(False)
                name = obj.name
                # # print("current selected object is:", bpy.context.selected_objects)
 
                bpy.ops.ed.undo_push()
                # repeat move&rotate until it won't cross the wall
                while True:
                    move_rotate(obj, T, T_0)

                    if check_if_cross_wall(obj):
                        bpy.ops.ed.undo()
                        obj = D.objects[name]
                        #print("***", "after undo object is ", obj.name)
                        check_if_cross_wall(obj)
                        #os.system("pause")
                        
                        bpy.context.view_layer.update()
                        continue

                    break
            
                count += 1
                accessableArea(obj)
            bpy.context.view_layer.update()
            new_c = total_cost(group)
            if i == 0:
                old_c = c_0
            if Metropolis(old_c, new_c, T):
                old_c = new_c
            else:
                bpy.ops.ed.undo()
                obj = D.objects[name]
            # print("Into next loop")
        T = T * alpha
        # print(T)
            

def Metropolis(old_c, new_c, T):
    """
    Description:
    
    """
    if new_c < old_c:
        
        return 1
    else:
        probability = math.exp((-1/T)*(new_c - old_c))
        # print(probability)
        # recieve new solution probably
        if random.random() < probability:
            return 1
        else:
            return 0   



if __name__ == "__main__":
    for object in bpy.data.collections["furniture"].all_objects:
        accessableArea(object)
    
    T_1 = time.time()
    simulated_annealing(bpy.data.collections["group_TV"])
    T_2 = time.time()
    print("Use ", (T_2 - T_1)*1000, "ms")