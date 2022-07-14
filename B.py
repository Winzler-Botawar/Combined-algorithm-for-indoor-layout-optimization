from curses.panel import bottom_panel
import bpy
import numpy as np
import math
import A

import bmesh
from bpy.props import BoolProperty, FloatVectorProperty
import mathutils
from bpy_extras import object_utils

import shapely
import shapely.geometry

C = bpy.context
D = bpy.data

room_lenth = room_width = 20

step_x, step_y = (50, 50)
door = [12, 0]
window = [37, 50]

energy_matrix = np.zeros(shape=(step_x, step_y))

def mesher(object):
    """
    Description: grid the floor of room, in this function, object means plane of floor.

    Input: C.object

    """
    
    width = object.dimensions[:][0]
    length = object.dimensions[:][1]

    x = np.linspace(0, width, step_x)
    y = np.linspace(0, length, step_y)

    X, Y = np.meshgrid(x, y)
    return X, Y

def center():
    """
    Description:

    This need you choose objects of one collection first, then it calculate center of these.

    Return: Point(x, y)
    """
    rectangle = group_bounding_box()
    return rectangle.centroid


def move(group, bottom_center, target_x, target_y):
    """
    Description:

    move group to target point

    """
    # create empty object
    delta_x = target_x - bottom_center.x
    delta_y = target_y - bottom_center.y
    
    # bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=(c.x, c.y, 0), scale=(1, 1, 1))
    # bpy.ops.object.collection_link(collection='')
    # attach group object to it

    for object in group.all_objects:
        object.location[0] += delta_x
        object.location[1] += delta_y

    

def rotate(group, angle):
    pass

def distance(point_1, point_2):
    """
    Description: give two points return their distance

    param: point_i: [x, y]
    """
    dis = math.sqrt(math.pow(point_2[0]-point_1[0], 2) + math.pow(point_2[1]-point_1[1], 2))

    return dis


def initial_energy_matrix(object, center, door, window):
    """
    Description:
    Create an energy_matrix which overlay the coordinate

    Return: initial energy matrix(2-dimensions)
    """
    
    X, Y = mesher(object)
    center = [X[int(step_x/2)][int(step_y/2)], Y[int(step_x/2)][int(step_y/2)]]

    for i in range(0, step_x, 1):
        for j in range(0, step_y, 1):
            d_1 = distance(door, [X[i][j], Y[i][j]])
            d_2 = distance(center, [X[i][j], Y[i][j]])
            d_3 = distance(window, [X[i][j], Y[i][j]])
            D_mean = 0.1 * d_1 + 0.3 * d_2 + 0.6 * d_3
            D_var = math.sqrt(math.pow(D_mean - 0.1*d_1, 2) + math.pow(D_mean - 0.3*d_2, 2) + math.pow(D_mean - 0.6*d_3, 2))

            energy_matrix[int(step_x - i - 1), j] = D_mean/D_var

    return energy_matrix

def check_if_hit(group):
    """
    Description:
    check if group out of wall

    Return: True or False
    """
    for obj in group.objects:
        obj.select_set(True)
    group_bottom_poly = group_bounding_box()
    for point in group_bottom_poly.exterior.coords:
        if point[0] > room_lenth or point[0] < 0 or point[1] > room_width or point[1] < 0:
            return True
        else:
            return False

def calc_energy_matrix():
    """
    Description:
    Calculate the energy matrix using given plane and occupied area list

    Return: energy matrix(2-dimensions)
    """

# from blender templates
def add_box(width, height, depth):
    """
    This function takes inputs and returns vertex and face arrays.
    no actual mesh data creation is done here.
    """

    verts = [(+1.0, +1.0, -1.0),
             (+1.0, -1.0, -1.0),
             (-1.0, -1.0, -1.0),
             (-1.0, +1.0, -1.0),
             (+1.0, +1.0, +1.0),
             (+1.0, -1.0, +1.0),
             (-1.0, -1.0, +1.0),
             (-1.0, +1.0, +1.0),
             ]

    faces = [(0, 1, 2, 3),
             (4, 7, 6, 5),
             (0, 4, 5, 1),
             (1, 5, 6, 2),
             (2, 6, 7, 3),
             (4, 0, 3, 7),
            ]

    # apply size
    for i, v in enumerate(verts):
        verts[i] = v[0] * width, v[1] * depth, v[2] * height

    return verts, faces

def group_bounding_box():
    minx, miny, minz = (999999.0,)*3
    maxx, maxy, maxz = (-999999.0,)*3
    location = [0.0,]*3
    for obj in bpy.context.selected_objects:
        for v in obj.bound_box:
            v_world = obj.matrix_world @ mathutils.Vector((v[0],v[1],v[2]))

            if v_world[0] < minx:
                minx = v_world[0]
            if v_world[0] > maxx:
                maxx = v_world[0]

            if v_world[1] < miny:
                miny = v_world[1]
            if v_world[1] > maxy:
                maxy = v_world[1]

            if v_world[2] < minz:
                minz = v_world[2]
            if v_world[2] > maxz:
                maxz = v_world[2]

    verts_loc, faces = add_box((maxx-minx)/2, (maxz-minz)/2, (maxy-miny)/2)
    mesh = bpy.data.meshes.new("BoundingBox")
    bm = bmesh.new()
    for v_co in verts_loc:
        bm.verts.new(v_co)

    bm.verts.ensure_lookup_table()

    for f_idx in faces:
        bm.faces.new([bm.verts[i] for i in f_idx])

    bm.to_mesh(mesh)
    mesh.update()
    location[0] = minx+((maxx-minx)/2)
    location[1] = miny+((maxy-miny)/2)
    location[2] = minz+((maxz-minz)/2)
    bbox = object_utils.object_data_add(bpy.context, mesh, operator=None)
    # does a bounding box need to display more than the bounds??
    bbox.location = location
    bbox.display_type = 'BOUNDS'
    bbox.hide_render = True

    mat = C.object.matrix_world
    verts = [mat @ v.co for v in C.object.data.vertices][:]
    bottom_points = [verts[i][:][0:2] for i in range(len(verts)) if verts[i][:][2] == 0]
    group_aabb_bottom_poly = shapely.geometry.MultiPoint(bottom_points).convex_hull
    bpy.ops.outliner.delete()

    return group_aabb_bottom_poly

def calc_occupied_energy(group_aabb_bottom_poly, X, Y):
    """
    Description: calculate total energy of area occupied by some group

    Return: total energy
    """
    occupied_area_energy = 0

    p_1 = group_aabb_bottom_poly.exterior.coords[0]
    p_2 = group_aabb_bottom_poly.exterior.coords[2]
    for i in range(0, step_x, 1):
        for j in range(0, step_y, 1):
            if (X[i][j] > p_1[0] and X[i][j] < p_2[0]) and (Y[i][j] > p_1[1] and Y[i][j] < p_2[1]):
                occupied_area_energy += energy_matrix[step_x - i - 1, j]

    return occupied_area_energy


def group_access_matrix(group):
    pass


# def init(group):
#     door = energy_matrix[12, 0]
#     window = [37, 50]

def update_energy_matrix():
    """
    Description:


    """

def set_group(group, energy_matrix):
    """
    Description:
    set group to the area where its energy matrix value max

    Param: group: collection of some furniture

    """

    for object in bpy.data.collections["furniture"].all_objects:
        object.select_set(False)

    # grid the floor plane
    bpy.data.objects['Plane'].select_set(True)
    X, Y = mesher(object)
    bpy.data.objects['Plane'].select_set(False)

    
    EM = energy_matrix
    
    # choose all objects, and move to highest energy point. 
    for obj in group.objects:
        obj.select_set(True)

    bottom_poly = group_bounding_box()
    bottom_center = bottom_poly.centroid

    max_x, max_y = np.unravel_index(EM.argmax(), EM.shape) 
    move(group, bottom_center, X[max_x][max_y], Y[max_x][max_y])





    # for group in furniture.children:
    #     for obj in group.objects:
    #         obj.select_set(True)
        
    #     group_bottom_poly = group_bounding_box()

    #     max_x, max_y = np.unravel_index(EM.argmax(), EM.shape) 
    #     move(group, X[max_x][max_y], Y[max_x][max_y])

    #     if check_if_hit(group):
    #         EM[max_x, max_y] = 0

    #     max_value = []
    #     for i in range(4):
    #         rotate(group, 90)
    #         max_value.append(calc_occupied_energy(group_bottom_poly, X, Y))
        
    #     if max_value.index(max(max_value)) == 0:
    #         bpy.ops.ed.undo()
    #         bpy.ops.ed.undo()
    #         bpy.ops.ed.undo()
    #     elif max_value.index(max(max_value)) == 1:
    #         bpy.ops.ed.undo()
    #         bpy.ops.ed.undo()
    #     elif max_value.index(max(max_value)) == 2:
    #         bpy.ops.ed.undo()
    #     elif max_value.index(max(max_value)) == 3:
    #         pass
        

