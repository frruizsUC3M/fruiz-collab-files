#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random

import argparse


def save_points_to_file(filename, points):

	with open(filename, 'w') as f:

		buff = '{}\n'.format(len(points))
		buff = '{}{}'.format(buff, ''.join(points))
		f.write(buff)


def subproblem_generator(filename, output_dir, min_points, steps=1):
	
	points = []

	with open(filename, 'r') as f:
		lines = f.readlines()
		points = lines[1:]

	while len(points) > min_points:

		filename_new_problem = os.path.join(output_dir, 'subproblem_{}'.format(len(points)))
		save_points_to_file(filename_new_problem, points)

		for i in range(steps):
			point_to_remove = random.randint(0, len(points))
			points = [p for i, p in enumerate(points) if i != point_to_remove]


if __name__ == '__main__':
	
	filename = 'examples/eil51.json'
	output_dir = 'examples/reduction_vs_exact'
	min_points = 5

	subproblem_generator(filename, output_dir, min_points, 2)