import sys
import os
import logging
import re
import functools
import fnmatch
import numpy as np
import torch.nn as nn


def extract_height(str1):
    height_list = [30, 60, 90, 120]
    numbers = list(map(lambda y:str(y), range(10)))

    height_list = list(map(lambda y:str(y), height_list))
    lenth1=len(str1)

    extracted_result = []
    for height in height_list:
        lenth2 = len(height)
        index_list = indexstr(str1, height)
        # print(index_list)
        index_result = []
        for index in index_list:
            if str1[index - 1] == '/': # at the beginning of the name
                if (str1[index + lenth2] == "m") or (str1[index + lenth2] == "mi"):
                    index_result.append(index)
            elif (str1[index - 1] not in numbers) and (
                    str1[index - 1] != '.') and (str1[index - 1] != ':') and (str1[index - 1] != '/'):  # the character before this word should not be a number
                if index + lenth2 < lenth1:
                    if (str1[index + lenth2] not in numbers): # the character after this word should not be a number
                        index_result.append(index)
                else:
                    index_result.append(index)
        if len(index_result) != 0:
            if len(index_result) == 1:
                extracted_result.append(str1[index_result[0]: index_result[0]+lenth2])
            else: # maybe because the same name appears twice in the path
                extracted_result.append(str1[index_result[0]: index_result[0] + lenth2])
                # print(str1)
                # for each in range(len(index_result)):
                #     print(str1[index_result[each]: index_result[each]+lenth2])
    if (len(extracted_result) == 0) or (len(extracted_result)) > 1 :
        return -1
    else:
        return int(extracted_result[0])



def indexstr(str1,str2):
    '''查找指定字符串str1包含指定子字符串str2的全部位置，
    以列表形式返回'''

    lenth2=len(str2)
    lenth1=len(str1)
    indexstr2=[]
    i=0
    while str2 in str1[i:]:
        indextmp = str1.index(str2, i, lenth1)
        indexstr2.append(indextmp)
        i = (indextmp + lenth2)
    return indexstr2
