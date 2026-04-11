"""
- Normalizes map with 95 percentile, to change the percentile value modify line number 27.

Author: Ashwin Dhakal
"""
import mrcfile
import numpy as np
import os
import argparse

#TODO: Automate voxel size acquisition, this was done as shrec data showing nan voxel spacing

def execute(inputs, user_default_voxel_size=1):
    count = 0
    map_names = [fn for fn in os.listdir(inputs) if not fn.endswith(".ent")]
    # print(map_names)
    
    for maps in range(len(map_names)):
        if map_names[maps] == "reconstruction.mrc":
            resample_map = map_names[maps]
            # print(resample_map)
            os.chdir(inputs)
            # print(inputs)
            # clean_map = mrcfile.open(resample_map, mode='r') #this doesnot work for tomogram
            clean_map = mrcfile.read(resample_map)
            # map_data = deepcopy(clean_map.data)  #original
            clean_map = np.array(clean_map.data)
            map_data = clean_map.copy()
            
            # normalize with percentile value
            print("### Normalizing Tomogram with 95-percentile for ", resample_map, " ###")
            try:
                percentile = np.percentile(map_data[np.nonzero(map_data)], 95)
                map_data /= percentile
            except IndexError as error:
                count += 1

            # set low valued data to 0
            print("### Setting all values < 0 to 0 for ", resample_map, " ###")
            map_data[map_data < 0] = 0
            print("### Setting all values > 1 to 1 for ", resample_map, " ###")
            map_data[map_data > 1] = 1
            
            # Get original voxel size
            # Handle potential NaN or zero values in voxel size
            if 'shrec_' in inputs:
                default_voxel_size = 1
            elif 'tomogram' in inputs:
                default_voxel_size = 1
            elif 'CryoETPortal' in inputs:
                default_voxel_size = 10
            elif 'MaxPlanck' in inputs:
                default_voxel_size = 10.2
            elif user_default_voxel_size is not None:
                default_voxel_size = user_default_voxel_size
            else:
                while True:
                    user_voxel_size = input(
                        f"Voxel size is not available for dataset folder '{inputs}'. "
                        "Please enter voxel size in angstroms: "
                    ).strip()
                    try:
                        default_voxel_size = float(user_voxel_size)
                        if default_voxel_size <= 0:
                            print("Voxel size must be a positive number. Please try again.")
                            continue
                        break
                    except ValueError:
                        print("Invalid voxel size input. Please enter a numeric value.")
            
            # Validate each dimension's voxel size
            x_voxel = default_voxel_size
            y_voxel = default_voxel_size
            z_voxel = default_voxel_size
                                
                
            with mrcfile.new(os.path.splitext(map_names[maps])[0] + "_normalized_map.mrc", overwrite=True) as mrc:
                mrc.set_data(map_data)
                mrc.voxel_size = (x_voxel, y_voxel, z_voxel)
                # mrc.header.origin = clean_map.header.origin
                mrc.close()
            print("### Wrote file to ", inputs, " ###")
    print("The number of non normalized index: ", count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize tomograms in each subfolder.")
    parser.add_argument(
        "--input-path",
        default="sample_input_data/tomogram_collection",
        help="Path to dataset root (relative to repository root by default).",
    )
    parser.add_argument(
        "--default-voxel-size",
        type=float,
        default=None,
        help="Optional fallback voxel size in angstroms for unknown dataset folder names.",
    )
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    input_path = os.path.abspath(os.path.join(repo_root, args.input_path))

    maps = [fn for fn in os.listdir(input_path) if not fn.endswith(".DS_Store")]
    for m in range(len(maps)):
        execute(os.path.join(input_path, maps[m]), args.default_voxel_size)
    print("Tomogram Normalization Complete!")