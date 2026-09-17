import numpy as np
import rasterio as rio
import os
from tqdm import tqdm
from pathlib import Path
from rasterio.merge import merge as rio_merge
import glob
from rasterio.windows import Window
from modules.config import Config

class AverageMerger:
	def __init__(self):
		self.sum = None
		self.count = None
	
	def merge(self, merged_data, new_data, merged_mask, new_mask, coff, roff, **kwargs):
		if self.sum is None:
			h, w = merged_data.shape[-2:]
			self.sum = np.zeros((h, w), dtype=np.uint16)
			self.count = np.zeros((h, w), dtype=np.uint8)

		valid = ~new_mask

		if valid.ndim == 3:
			valid = valid[0]
			data = new_data[0]
		else:
			data = new_data
		
		if not valid.any():
			return

		rows, cols = np.nonzero(valid)
		rmin, rmax = roff + rows.min(), roff + rows.max() + 1
		cmin, cmax = coff + cols.min(), coff + cols.max() + 1

		lr, lc = slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1)

		print(lr, lc, rmin, rmax, cmin, cmax)

		self.sum[rmin:rmax, cmin:cmax] += np.where(valid[lr, lc], data[lr, lc], 0)
		self.count[rmin:rmax, cmin:cmax] += valid[lr, lc].astype(np.uint8)

class Postprocessor:
	def __init__(self, input_dir: str, image_extension: str = ".tif"):
		self.input_dir = Path(input_dir)
		self.image_extension = "." + image_extension.lstrip(".").lower()

#	def create_alpha_sources(self, img_paths):
		#mem_files = []
		#alpha_sources = []
		#print("Creating alpha masks...")

		#for path in tqdm(img_paths):
			#with rio.open(path) as src:
				#alpha_profile  = src.profile.copy()
				#alpha_profile.update(count=1)

				#memfile =  rio.MemoryFile()
				#ds = memfile.open(**alpha_profile)
				#mask = src.dataset_mask()
				#ds.write(np.expand_dims(mask, 0))
				#alpha_sources.append(ds)
				#mem_files.append(memfile)
				##memfile.close()
		#return alpha_sources, mem_files

	def clip(self, img, mask, meta):
		rows, cols = np.where(mask > 0)
		row_min, row_max = rows.min(), rows.max() + 1
		col_min, col_max = cols.min(), cols.max() + 1
    
		clipped = img[row_min:row_max, col_min:col_max]
		clipped_mask = mask[row_min:row_max, col_min:col_max]


		a, b, c, d, e, f = meta["transform"][:6]
    
		meta.update({
			"transform": rio.Affine(a, b, c + col_min * a, d, e, f + row_min * e),
			"height": row_max - row_min,
			"width": col_max - col_min
		})
    
		return clipped, clipped_mask, meta

	def __call__(self, output_path: str = "./", output_type="masks", threshold=127):
		print(os.path.join(self.input_dir, f"*{self.image_extension}"))

		img_paths = glob.glob(os.path.join(self.input_dir, f"*{self.image_extension}"))
		unique_stems = set(["_".join(Path(x).name.split("_")[:-2]) for x in img_paths])
		os.makedirs(output_path, exist_ok=True)

		if not img_paths:
			return

		with rio.open(img_paths[0], "r") as f:
			profile = f.profile.copy()

		for unique in tqdm(unique_stems):
			img_paths = glob.glob(os.path.join(self.input_dir, f"{unique}*{self.image_extension}"))

			if not img_paths:
				continue

			print(f"Merging {unique}...({len(img_paths)} images)")

			#alpha_sources, mem_files = self.create_alpha_sources(img_paths)
			#mask, transform = rio_merge(alpha_sources, method="max")
			
			#print("Finished merging masks...")

			#merger = AverageMerger()
			#_, transform = rio_merge(img_paths, method=merger.merge)

			count, transform = rio_merge(img_paths, method="count", dtype=np.uint8)
			total, _          = rio_merge(img_paths, method="sum",   dtype=np.uint16)

			c = count[0] if count.ndim == 3 else count
			s = total[0] if total.ndim == 3 else total


			avg = np.zeros(s.shape, dtype=np.uint8)
			np.floor_divide(s, c, out=avg, where=c > 0)

			if output_type == "masks":
				avg = np.where(avg >= threshold, np.uint8(255), np.uint8(0))

			mask = (c > 0).astype(np.uint8) * 255


			profile.update({
					"width": s.shape[1],
					"height": s.shape[0],
					"transform": transform,
					"count": 1,
					"compression": "LZW",
					"dtype": rio.uint8,
					"nodata": None
				})

			out_filename = os.path.join(output_path, f"{unique}{self.image_extension}")
			#memfile = rio.MemoryFile()
			#dst = memfile.open(**profile)
			#avg = np.nan_to_num(avg, nan=0, posinf=255, neginf=0)
			#dst.write(np.expand_dims(avg, 0))
			#dst.write_mask(mask.squeeze())

			print(f"Clipping {unique} by mask...")

			clipped, clipped_mask, meta = self.clip(avg, mask, profile)
			clipped[clipped < threshold] = 0


	

			with rio.open(out_filename, "w", **meta) as f:
				f.write(clipped, 1)
				f.write_mask(clipped_mask)

			
			#dst.close()
			#memfile.close()

def postprocess(config):
	postprocessor = Postprocessor(config.image_dir, config.image_extension)
	postprocessor(config.output_dir, config.output_type, config.threshold)


if __name__ == "__main__":
	config = Config.from_yaml("config2.yaml")
	postprocess(config.postprocessor)