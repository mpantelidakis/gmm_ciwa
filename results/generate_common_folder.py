import os,cv2, sys
import numpy as np
import results_utils 
import helpers

crop_height = 320
crop_width = 480
dataset_path = '../Testset'
# Get the names of the classes so we can record the evaluation results
print("Retrieving dataset information ...")
class_names_list, label_values = helpers.get_label_info("./class_dict.csv")
# print(class_names_list, label_values)
class_names_string = ""
for class_name in class_names_list:
    if not class_name == class_names_list[-1]:
        class_names_string = class_names_string + class_name + ", "
    else:
        class_names_string = class_names_string + class_name

num_classes = len(label_values)


test_input_names, test_output_names = results_utils.prepare_data(dataset_path)
# Create directories if needed
if not os.path.isdir("%s"%("Test_GMM")):
        os.makedirs("%s"%("Test_GMM"))

# Run testing on ALL test images
for ind in range(len(test_input_names)):
    sys.stdout.write("\rRunning test image %d / %d"%(ind+1, len(test_input_names)))
    sys.stdout.flush()

    gmm_prediction = np.expand_dims(np.float32(results_utils.load_image(test_input_names[ind])[:crop_height, :crop_width]),axis=0)/255.0
    gt = results_utils.load_image(test_output_names[ind])[:crop_height, :crop_width]
    gt = helpers.reverse_one_hot(helpers.one_hot_it(gt, label_values))

    gmm_prediction = np.array(gmm_prediction[0,:,:,:])
    gmm_prediction = helpers.reverse_one_hot(gmm_prediction)
    out_vis_image = helpers.colour_code_segmentation(gmm_prediction, label_values)

    file_name = results_utils.filepath_to_name(test_input_names[ind])
   
    gt = helpers.colour_code_segmentation(gt, label_values)

    cv2.imwrite("%s/%s_pred.png"%("Test_GMM", file_name),cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))
    cv2.imwrite("%s/%s_gt.png"%("Test_GMM", file_name),cv2.cvtColor(np.uint8(gt), cv2.COLOR_RGB2BGR))


