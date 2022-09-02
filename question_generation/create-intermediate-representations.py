#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 10:21:37 2022

@author: savitha
"""

import json
import argparse

def create_clause(name, state, shape,color,size,material,variable):
    clause = "{}({},{},{},{},{},{}{})".format(name, state, shape,color,size,material,name[0],variable)
    return clause

def generate(query, step, var, program):
    instr = ""
    command = program[step]
    type_ = command['type']
    inputs = command['inputs']
    val_inp = command['value_inputs']

    is_filter = type_.startswith('filter')
    is_intermediate = len(inputs)==1 and inputs[0] != 0
    is_first_inter = len(inputs)==1 and inputs[0] == 0
    is_first = len(inputs)==0

    #instr+instr1+type_.split("_")[1]+val_inp[0]
    if type_ == 'scene':
        instr = "shape=objects"
        return [instr,var]
    if is_filter:
        if is_first:
            ins = instr
            target = "shape=objects,"
            val = val_inp[0]
            if type_ == 'filter_color':
                target = "shape=objects,color="
            elif type_ == 'filter_shape':
                target = target.replace(',','')
                val = ""
            else:
                [_,var] = generate(query,inputs[0], var, program)
            if type_ == 'filter_material' or 'filter_size':
                ins = ""
            instr = ins + target + val
            return [instr,var]
        if is_intermediate and not type_ == 'filter_shape':
            [instr1,var1] = generate(query,inputs[0], var, program)
            instr = instr+instr1+",%s=".format(type_.split("_")[1])+val_inp[0]
            return [instr,var1]
        if type_ == 'filter_shape' and is_first_inter:
            instr = "shape="+val_inp[0]
            return [instr,var]
    elif type_ == 'count':

        [instr1,var1] = generate(query,inputs[0],  var, program)
        instr1a = instr1.split(',')
        shape = "_"
        size = "_"
        color = "_"
        material="_"
        for args in range(len(instr1a)):
            arg_first = instr1a[args]
            attr_val = arg_first.split("=")
            if attr_val[0] == "shape":
                shape = shape+attr_val[1]
            elif attr_val[0] == "size":
                size = size+attr_val[1]
            elif attr_val[0] == "color":
                color = color+attr_val[1]
            else:
                material = material+attr_val[1]

        state = 0
        instr = create_clause("count", state, shape,color,size,material,var1)
        return [instr,var1+1]
    elif type_ == "subtraction":
        first = inputs[0]
        second = inputs[1]
        [instr1,var1] = generate( query, first,var, program)
        [instr2,var2] = generate( query, second,var1, program)
        import ipdb; ipdb.set_trace()
        instr2d = "decrease("+instr2+", 0)"
        inter = instr1+","+instr2d+","+query
        return [inter,var2]

def question_type_to_query(instance,num_states):
    # TODO Ask about more than only shape
    question = instance['question'].split('.')[-1]
    query_object = "objects"

    if 'balls' in question or 'spheres' in question:
        query_object = "sphere"
    elif 'blocks' in question or 'cubes' in question:
        query_object = "cube"
    elif 'cylinders' in question:
        query_object = "cylinder"

    query = "count({},{},_,_,_,?X)".format(num_states,query_object)

    return query

def get_question_type(instance):
    qtype = ""

    return qtype

def create_intermediate(instance):
    num_states = len(instance['question'].split('.'))
    query = question_type_to_query(instance, num_states)
    q_type = get_question_type(instance)

    program = instance['program']
    [inter,var1] = generate(query, len(program)-1, 1,  program)
    return inter

if __name__=='__main__':
    parser = argparse.ArgumentParser("Parsing command line arguments to CLEVR-math intermediate representation generator")
    parser.add_argument('--json_file', type=str, help='JSON file to generate intermediate representations for')
    args = parser.parse_args()

    # Opening JSON file
    f = open(args.json_file)
    data = json.load(f)

    # Iterating through the json list
    json_object={}
    q_updated = []
    #Generate intermediate representation and equation
    for instance in data['questions']:
        instance['inter'] = create_intermediate(instance)
        q_updated.append(instance)

    # Closing file
    f.close()
    json_object['info'] = data['info']
    json_object['questions'] = q_updated
    #json_object = json.parse(json.stringify(json_object));
    print(json_object)
    #a_file = open("/home/savitha/Documents/MWPS/CLEVR-Math/subtest_inter.json", "w")
    #json.dump(json_object, a_file)
    #a_file.close()
