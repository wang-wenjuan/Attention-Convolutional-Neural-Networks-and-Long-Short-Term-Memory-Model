import csv
import pickle
import cv2
from cv_utils import get_regions, extract_colors, img_3d_to_2d_vector, fvc_class_map, lulc_class_map, rsei_class_map
import numpy as np

"""
统计每个地区各级占比
"""

region_dict = {
    0: 'Tekes-Zhaosu Subregion',
    1: 'South Bank of the Ili River',
    2: 'North Bank of the Ili River'
}

map_class_dicts = [fvc_class_map, rsei_class_map, lulc_class_map]

# for type_num, map_type in enumerate(map_types):
#     print(50*'=')
#     print(map_type, ':')
#     # 读取图像
#     image = cv2.imread(f"../data/processed/{map_type}_predicted_map.png")
#
#     regions = get_regions(image)
#     for region_num, region in enumerate(regions):
#         print('\t', region_dict[region_num], ':')
#         # color = extract_colors(region)
#         # cv2.namedWindow("Win", cv2.WINDOW_NORMAL)
#         # cv2.resizeWindow("Win", 800, 600)
#         # cv2.imshow('Win', region)
#         # cv2.waitKey()
#
#         region_2d = img_3d_to_2d_vector(region, map_type)
#         counts = np.bincount(region_2d.flatten())
#         # print(counts)
#         for i, v in enumerate(counts[1:]):
#             print('\t\t', map_class_dicts[type_num][i+1])
#             print('\t\t\t', v / sum(counts[1:]) * 100, '%')
#         # for count in enumerate(counts):
#         #     print(f"值 {i} 出现次数: {count}")
#         # cv2.imwrite('../data/interim/colors.png', color)
#         # print(color)

with open('../data/processed/region_statistics.csv', mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)

    # Write header
    writer.writerow(['Map Type', 'Region', 'Class', 'Percentage'])

    # Loop over each map type and region, calculate, and write statistics
    for type_num, map_type in enumerate(map_types):
        print(50 * '=')
        print(map_type, ':')

        # Read the image
        image = cv2.imread(f"../data/processed/{map_type}_predicted_map.png")

        regions = get_regions(image)
        for region_num, region in enumerate(regions):
            print('\t', region_dict[region_num], ':')

            # Process each region
            region_2d = img_3d_to_2d_vector(region, map_type)
            counts = np.bincount(region_2d.flatten())

            for i, v in enumerate(counts[1:]):  # Start from 1 to skip background (if present)
                class_name = map_class_dicts[type_num][i + 1]
                percentage = v / sum(counts[1:]) * 100

                # Print to console (optional, for debugging)
                print('\t\t', class_name)
                print('\t\t\t', percentage, '%')

                # Write row to CSV
                writer.writerow([map_type, region_dict[region_num], class_name, percentage])

print("CSV file created successfully at '../data/processed/region_statistics.csv'")