"""
Generate (and save) a mapping from coco image id to fc7 feature vectors
python generateIdToFc7Mappings.py

Saves the H5 file in : /srv/share/samyak/visdial-pool/data/visdial-v0.9-fc7-features.h5
"""

import json
import argparse
import h5py as h5
import numpy as np

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"-featuresH5",
		help="h5 data file containing the fc7 descriptors of coco images in VisDial v0.9",
		default="/srv/share/abhshkdz/data/visdial/data_img_relu7.h5"
	)
	parser.add_argument(
		"-idsJson",
		help="JSON file containing the order in which the descriptors have been saved in -featuresH5",
		default="/srv/share/abhshkdz/data/visdial/visdial_params.json"
	)
	parser.add_argument(
		"-outputH5",
		help="Output H5 file to save coco image Id to fc7 feature vector",
		default="/srv/share/samyak/visdial-pool/data/visdial-v0.9-fc7-features.h5"
	)
	args = parser.parse_args()

	ids = json.load(open(args.idsJson, "r"))
	train_ids, val_ids = ids['unique_img_train'], ids['unique_img_val']

	fc7 = h5.File(args.featuresH5, "r")

	idToFc7 = dict()
	print ("\nBuilding dict() for train images...")
	for i in range(len(fc7['images_train'])):
		idToFc7[train_ids[i]] = np.asarray(fc7['images_train'][i])
	
	print ("\nBuilding dict() for val images...")
	for i in range(len(list(fc7['images_val']))):
		idToFc7[val_ids[i]] = np.asarray(fc7['images_val'][i])
	
	with h5.File(args.outputH5, "w") as f:
		for id, descr in idToFc7.items():
			f.create_dataset(str(id), data=np.asarray(descr))