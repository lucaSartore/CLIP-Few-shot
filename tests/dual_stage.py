from prompt_learner import *
import torch as pt
def main():


    u = Util()
    u.build_model(pt.Tensor(), pt.Tensor())


    test_dataset = get_data_splitted("RN50", u.device)[2]

    first_stage_prompts = u.load_first_stage_prompts() 
    
    for image, label in test_dataset:
        image = image.to(u.device).unsqueeze(dim=0)

        best_prompt_index = u.inference(image, first_stage_prompts)

        second_stage_prompts = u.load_second_stage_prompts( best_prompt_index)

        predicted_label = u.inference(image, second_stage_prompts)

        print(f"Predicted label: {predicted_label}, actual label: {label}")


if __name__ == "__main__":
    main()



