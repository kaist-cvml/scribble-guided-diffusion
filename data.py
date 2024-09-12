import os
import cv2
import torch
import numpy as np

from PIL import Image
from torch.nn import functional as F

    
def get_bbox_from_scribble(scribble):
    if torch.sum(scribble) == 0:
        return torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1)

    y_indices, x_indices = torch.where(scribble)
    
    x_min = torch.min(x_indices)
    x_max = torch.max(x_indices)
    y_min = torch.min(y_indices)
    y_max = torch.max(y_indices)

    return x_min, y_min, x_max - x_min, y_max - y_min


class PromptInput:
    def __init__(
        self,
        batch_size,
        prompts, 
        stroke_dirs=None,
        output_dirs=None, 
        save_scribble_dirs=None,
        save_mask_dirs=None, 
        vis_dirs=None,
        scribble_res=64,
        bbox_padding=0.05,
        max_num_masks=30,
        device=torch.device('cpu')
    ):
        self.batch_size = batch_size
        self.prompts = prompts
        self.stroke_dirs = stroke_dirs
        
        self.phrases = None
        self.gligen_phrases = None
        self.tokens = None
        self.phrase_indices = None

        # lists
        self.scribble_list = None
        self.bbox_list = None
        
        # grounding inputs
        self.grounding_inputs = None
        self.max_num_masks = max_num_masks
        
        # tensors per token
        self.scribbles = None
        self.masks = None
        self.token_indices = None
        
        # tensors per individual
        self.individual_scribbles = None
        self.individual_masks = None
        
        self.phrase_to_obj = None
        self.obj_to_phrase = None
        self.phrase_to_individual = None
        self.individual_to_phrase = None
        
        self.stroke_paths = None
        self.max_num_tokens = 0
        self.max_num_individuals = 0
        
        self.bbox_padding = bbox_padding
        self.scribble_res = scribble_res
        
        self.output_dirs = output_dirs
        self.save_scribble_dirs = save_scribble_dirs
        self.save_mask_dirs = save_mask_dirs
        self.vis_dirs = vis_dirs
        
        self.device = device
        
        return
    
    dir_attributes = ['output_dirs', 'save_scribble_dirs', 'save_mask_dirs', 'vis_dirs']
    
    def prompt_to_tokens(self, prompt, tokenizer):
            tokens = tokenizer.encode(prompt)
            decoder = tokenizer.decode

            prompt_tokens = [decoder(token) for token in tokens]
            return prompt_tokens
    
    def valid_check(self, exp_mode=False):
        def initialize_dir(attr_name):
            attr = getattr(self, attr_name)
            if attr is not None and isinstance(attr, str):
                setattr(self, attr_name, [attr] * self.batch_size)
        
        def verify_and_create_dir(dir_list):
            for dir_path in dir_list:
                os.makedirs(dir_path, exist_ok=True)

        for attr in PromptInput.dir_attributes:
            initialize_dir(attr)

        if isinstance(self.prompts, str):
            self.prompts = [self.prompts] * self.batch_size

        assert len(self.prompts) == self.batch_size, "Number of prompts does not match batch size."

        if not exp_mode:
            for strokes_path in self.stroke_dirs:
                assert os.path.exists(strokes_path), f"Please specify the valid path to the input directory: {strokes_path}"

        for attr in PromptInput.dir_attributes:
            dir_list = getattr(self, attr)
            verify_and_create_dir(dir_list)
            
        return
            
    def get_phrases_and_strokes_from_inputs(self):
        self.phrases = [None] * self.batch_size
        self.phrase_indices = [None] * self.batch_size
        self.stroke_paths = [None] * self.batch_size
        
        for batch in range(self.batch_size):
            self.phrases[batch] = []
            self.stroke_paths[batch] = []
            input_filenames = []
            input_indices = []

            input_files = os.listdir(self.stroke_dirs[batch])
            
            assert len(input_files) < 30, "Too many strokes in the input directory. Please limit the number of strokes to 30."
            
            for file in input_files:
                if not file.endswith('.jpg') and not file.endswith('.png'):
                    continue
                
                filename = os.path.splitext(file)[0]
                
                if '_' not in filename:
                    continue
            
                file_index = filename.split('_')[-1]
                filename = '_'.join(filename.split('_')[:-1]).replace('_', ' ')
 
                input_filenames.append(filename)
                input_indices.append(int(file_index))
            
            # get phrases by sorting input filenames based on their indices in prompt
            self.phrase_indices[batch] = sorted(input_indices)
            sorted_input_filenames = []
            
            for input_index in self.phrase_indices[batch]:
                for input_filename, input_file_index in zip(input_filenames, input_indices):
                    if input_index == input_file_index:
                        sorted_input_filenames.append(input_filename)
                        break
            
            for input_filename, input_index in zip(sorted_input_filenames, self.phrase_indices[batch]):
                self.phrases[batch].append(input_filename)
                self.stroke_paths[batch].append(os.path.join(self.stroke_dirs[batch], input_filename.replace(' ', '_') + '_' + str(input_index)))

        return
    
    
    def update_scribbles_and_phrases(self, prompt, scribbles_dict, text_encoder):
        '''
        Update scribbles and phrases from dataset in experiment.
        '''
        self.prompts = [prompt] * self.batch_size
        self.phrases = [None] * self.batch_size
        self.scribble_list = [None] * self.batch_size
        self.phrase_indices = [None] * self.batch_size
        
        for batch in range(self.batch_size):
            self.phrases[batch] = []
            self.scribble_list[batch] = []
            self.phrase_indices[batch] = []
            
            tokens = self.prompt_to_tokens(self.prompts[batch], text_encoder.tokenizer)
            
            phrases_with_indices = []
            
            for classname, scribble in scribbles_dict.items():
                tokenized_classnames = self.prompt_to_tokens(classname, text_encoder.tokenizer)[1:-1]
                
                phrases_with_indices += [(' '.join(tokenized_classnames), tokens.index(tokenized_classnames[0]), scribble)]
                
            sorted_phrases_with_indices = sorted(phrases_with_indices, key=lambda x: x[1])
            
            for phrase_with_index in sorted_phrases_with_indices:
                phrase, index, scribble = phrase_with_index
                self.phrases[batch].append(phrase)
                self.phrase_indices[batch].append(index)
                
                scribble = Image.fromarray(scribble.detach().numpy().astype(np.uint8)[0])
                scribble = scribble.convert('L')
                scribble = np.array(scribble)
                scribble = np.where(scribble < 128, 0, 255).astype(np.float32)
                scribble /= 255.
                self.scribble_list[batch].append(torch.from_numpy(scribble).float())
            
            print(self.prompts[batch], flush=True)
            print(self.phrases[batch], flush=True)
            print(self.phrase_indices[batch], flush=True)
            print(len(self.scribble_list[batch]), flush=True)
            
        return
        

    def get_scribbles_from_strokes(self, save_strokes=True, save_stroke_res=512):
        size = self.scribble_res
        self.scribble_list = [None] * self.batch_size
        kernel = np.ones((3, 3), np.uint8)
        
        for batch in range(self.batch_size):
            self.scribble_list[batch] = [None] * len(self.stroke_paths[batch])
            zero_strokes = np.full((save_stroke_res, save_stroke_res), 255, dtype=np.uint8)
            
            for i, stroke_path in enumerate(self.stroke_paths[batch]):
                
                if os.path.exists(stroke_path + '.png'):
                    stroke_path = stroke_path + '.png'
                    
                if os.path.exists(stroke_path + '.jpg'):
                    stroke_path = stroke_path + '.jpg'
                    
                strokes = Image.open(stroke_path).convert('L')
                strokes = np.array(strokes)
                strokes = 255 - strokes
                strokes = np.where(strokes < 128, 0, 255).astype(np.uint8)
                
                strokes = cv2.dilate(strokes, kernel, iterations=1)
                
                scribble = torch.from_numpy(strokes)
                scribble = scribble.float() / 255.
                scribble_resized = F.interpolate(
                    scribble.unsqueeze(0).unsqueeze(0), 
                    size=(size, size),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).squeeze(0)
                
                self.scribble_list[batch][i] = scribble_resized.bool().float().to(self.device)
                
                if save_strokes:
                    resized_strokes = cv2.resize(strokes, (save_stroke_res, save_stroke_res))
                    zero_strokes = np.where(resized_strokes == 255, 0, zero_strokes)
                
            if save_strokes:
                strokes = Image.fromarray(zero_strokes.astype(np.uint8))
                strokes_path = os.path.join('/'.join(self.stroke_paths[batch][0].split('/')[:-1]), 'strokes.png')
                strokes.save(strokes_path)
                

    def get_tensors_from_lists(self, text_encoder):
        self.tokens = [None] * self.batch_size
        for batch in range(self.batch_size):
            self.tokens[batch] = self.prompt_to_tokens(self.prompts[batch], text_encoder.tokenizer)

        token_indices_list = [None] * self.batch_size
        
        # tensors per token
        scribble_tensors = [None] * self.batch_size
        mask_tensors = [None] * self.batch_size
        
        # tensors per individual
        individual_scribble_tensors = [None] * self.batch_size
        individual_mask_tensors = [None] * self.batch_size
        
        self.obj_to_phrase = [None] * self.batch_size
        self.phrase_to_obj = [None] * self.batch_size
        self.phrase_to_individual = [None] * self.batch_size
        self.individual_to_phrase = [None] * self.batch_size
        self.bbox_list = [None] * self.batch_size
        self.gligen_phrases = [None] * self.batch_size
        
        for batch in range(self.batch_size):
            token_indices_list[batch] = []
            scribble_tensors[batch] = []
            mask_tensors[batch] = []
            
            individual_scribble_tensors[batch] = []
            individual_mask_tensors[batch] = []
           
            self.obj_to_phrase[batch] = []
            self.phrase_to_obj[batch] = []
            self.phrase_to_individual[batch] = []
            self.individual_to_phrase[batch] = []
            
            self.bbox_list[batch] = []
            self.gligen_phrases[batch] = []
            num_individuals = 0
            num_tokens = 0

            for phrase_index, phrase in enumerate(self.phrases[batch]):
                phrase_tokens = phrase.split(' ')
                # phrase_tokens = prompt_to_tokens(phrase, text_encoder.tokenizer)[1:-1]
                size = self.scribble_res
                start = self.phrase_indices[batch][phrase_index]
                
                self.phrase_to_obj[batch].append([])
                self.phrase_to_individual[batch].append([])
        
                for idx, phrase_token in enumerate(phrase_tokens):
                    # if phrase_token in {"a", "an", "the"}:
                    #     continue
                    
                    token_indices_list[batch].append(start + idx)
                    scribble_tensors[batch].append(self.scribble_list[batch][phrase_index])
                    
                    self.obj_to_phrase[batch].append(phrase_index)
                    self.phrase_to_obj[batch][phrase_index].append(num_tokens)
                    num_tokens += 1

                    
                scribble_npy = self.scribble_list[batch][phrase_index].cpu().numpy() * 255
                num_labels, labels_im = cv2.connectedComponents(scribble_npy.astype(np.uint8))
                
                phrase_mask = torch.zeros_like(self.scribble_list[batch][phrase_index]).to(self.device)
            
                for label in range(1, num_labels):
                    self.individual_to_phrase[batch].append(phrase_index)
                    self.phrase_to_individual[batch][phrase_index].append(num_individuals)
                    num_individuals += 1
                    
                    individual_scribble = np.where(labels_im == label, 1, 0).astype(np.uint8)
                    individual_scribble = torch.from_numpy(individual_scribble).float()
                    individual_mask = torch.zeros_like(individual_scribble).to(self.device)
                    
                    x_min, y_min, w, h = get_bbox_from_scribble(individual_scribble)
                    
                    x_max = x_min + w
                    y_max = y_min + h
                    
                    x_min = max(0, x_min - size * self.bbox_padding) / size
                    y_min = max(0, y_min - size * self.bbox_padding) / size
                    x_max = min(size, x_max + size * self.bbox_padding) / size
                    y_max = min(size, y_max + size * self.bbox_padding) / size
                    
                    self.bbox_list[batch].append([x_min, y_min, x_max, y_max])
                    
                    individual_scribble_resized = F.interpolate(
                        individual_scribble.unsqueeze(0).unsqueeze(0), 
                        size=(size, size),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0).squeeze(0).bool().float().to(self.device)
                    
                    individual_scribble_tensors[batch].append(individual_scribble_resized)
                    individual_mask[int(y_min * size):int(y_max * size), int(x_min * size):int(x_max * size)] = 1
                    individual_mask_tensors[batch].append(individual_mask)
                    phrase_mask = torch.max(phrase_mask, individual_mask)
                
                mask_tensors[batch].extend([phrase_mask] * len(phrase_tokens))
                for phrase_token in phrase_tokens:
                    self.gligen_phrases[batch].append(phrase_token)
                
            self.max_num_tokens = max(self.max_num_tokens, num_tokens)
            self.max_num_individuals = max(self.max_num_individuals, num_individuals)
            
            scribble_tensors[batch] = torch.stack(scribble_tensors[batch], dim=0)
            mask_tensors[batch] = torch.stack(mask_tensors[batch], dim=0)
            
            individual_scribble_tensors[batch] = torch.stack(individual_scribble_tensors[batch], dim=0)
            individual_mask_tensors[batch] = torch.stack(individual_mask_tensors[batch], dim=0)
        
        self.token_indices = torch.zeros((self.batch_size, self.max_num_tokens), dtype=torch.long).to(self.device)
        self.scribbles = torch.zeros((self.batch_size, self.max_num_tokens, self.scribble_res, self.scribble_res)).to(self.device)
        self.masks = torch.zeros((self.batch_size, self.max_num_tokens, self.scribble_res, self.scribble_res)).to(self.device)
        
        self.individual_scribbles = torch.zeros((self.batch_size, self.max_num_individuals, self.scribble_res, self.scribble_res)).to(self.device)
        self.individual_masks = torch.zeros((self.batch_size, self.max_num_individuals, self.scribble_res, self.scribble_res)).to(self.device)
        
        for batch in range(self.batch_size):
            self.token_indices[batch, :len(token_indices_list[batch])] = torch.tensor(token_indices_list[batch]).long().to(self.device)
            self.scribbles[batch, :len(token_indices_list[batch])] = scribble_tensors[batch]
            self.masks[batch, :len(token_indices_list[batch])] = mask_tensors[batch]
            
            self.individual_scribbles[batch, :len(individual_scribble_tensors[batch])] = individual_scribble_tensors[batch]
            self.individual_masks[batch, :len(individual_mask_tensors[batch])] = individual_mask_tensors[batch]
        
        return
    
    def update_token_tensors(self):
        updated_boxes = torch.zeros(self.batch_size, self.max_num_masks, 4).to(self.device)
        updated_masks = torch.zeros_like(self.masks).to(self.device)
        updated_scribbles = torch.zeros_like(self.scribbles).to(self.device)
        
        for batch in range(self.batch_size):
            for i in range(len(self.bbox_list[batch])):
                individual_to_phrase = self.individual_to_phrase[batch][i]
                phrase_to_tokens = self.phrase_to_obj[batch][individual_to_phrase]
                
                x_min, y_min, w, h = get_bbox_from_scribble(self.individual_scribbles[batch, i])
                
                x_max = x_min + w
                y_max = y_min + h
                
                x_min = max(0, x_min - self.scribble_res * self.bbox_padding) / self.scribble_res
                y_min = max(0, y_min - self.scribble_res * self.bbox_padding) / self.scribble_res
                x_max = min(self.scribble_res, x_max + self.scribble_res * self.bbox_padding) / self.scribble_res
                y_max = min(self.scribble_res, y_max + self.scribble_res * self.bbox_padding) / self.scribble_res
                
                self.individual_masks[batch, i] = torch.zeros_like(self.individual_masks[batch, i])
                self.individual_masks[batch, i, int(y_min * self.scribble_res):int(y_max * self.scribble_res),
                            int(x_min * self.scribble_res):int(x_max * self.scribble_res)] = 1
                
                for j in phrase_to_tokens:
                    updated_scribbles[batch, j] = torch.max(updated_scribbles[batch, j], self.individual_scribbles[batch, i])
                    updated_masks[batch, j, int(y_min * self.scribble_res):int(y_max * self.scribble_res),
                                    int(x_min * self.scribble_res):int(x_max * self.scribble_res)] = 1
                
                # # update mask
                # self.masks[batch, i] = torch.zeros_like(self.masks[batch, i])
                # self.masks[batch, i, int(y_min * self.scribble_res):int(y_max * self.scribble_res), 
                #            int(x_min * self.scribble_res):int(x_max * self.scribble_res)] = 1
                
                # self.masks[batch, i] = self.masks[batch, i].bool().float()
                self.bbox_list[batch][i] = [x_min, y_min, x_max, y_max]
                updated_boxes[batch, i] = torch.tensor(self.bbox_list[batch][i])
                
        self.masks = updated_masks
        # self.grounding_inputs['boxes'] = updated_boxes

        return
    
    def update_grounding_input(self, boxes, masks, text_embeddings, grounding_tokenizer_input):
        grounding_inputs = {
            "boxes": boxes,
            "masks": masks,
            "text_embeddings": text_embeddings
        }
        
        self.grounding_inputs = grounding_tokenizer_input.prepare(grounding_inputs)
        return
    
    
    def get_grounding_input(self, clip_model, clip_processor, grounding_tokenizer_input):
        def get_text_clip_feature(clip_model, clip_processor, phrase):
            txt_embeds = None

            if phrase is not None:
                inputs = clip_processor(text=phrase, return_tensors="pt", padding=True)
                inputs['input_ids'] = inputs['input_ids'].to(self.device)
                inputs['pixel_values'] = torch.ones(1, 3, 224, 224).to(self.device)
                inputs['attention_mask'] = inputs['attention_mask'].to(self.device)

                outputs = clip_model(**inputs)
                txt_embeds = outputs['text_model_output']['pooler_output']

            return txt_embeds
        
        # This is implemented for positonal encoding in GLIGEN.
        boxes = torch.zeros(self.batch_size, self.max_num_masks, 4).to(self.device)
        masks = torch.zeros(self.batch_size, self.max_num_masks).to(self.device)
        txt_embs = torch.zeros(self.batch_size, self.max_num_masks, 768).to(self.device)

        for batch in range(self.batch_size):
            txt_emb_list = []
            for i, bbox in enumerate(self.bbox_list[batch]):
                phrase = self.phrases[batch][self.individual_to_phrase[batch][i]]
                txt_emb = get_text_clip_feature(clip_model, clip_processor, phrase)
                txt_emb_list.append(txt_emb)
            
            idx = 0
            for bbox, txt_emb in zip(self.bbox_list[batch], txt_emb_list):
                boxes[batch, idx] = torch.tensor(bbox)
                masks[batch, idx] = 1

                if txt_emb is not None:
                    txt_embs[batch, idx] = txt_emb
                
                idx += 1

        self.update_grounding_input(boxes, masks, txt_embs, grounding_tokenizer_input)
        
        return
    
    
    def save_scribbles(self, save_individual=False, save_scribble_dir=None, timestep=1000):
        save_scribble_dir = save_scribble_dir if save_scribble_dir is not None else self.save_scribble_dirs[0]
        
        if save_individual:
            save_scribble_dir = os.path.join(save_scribble_dir, 'individuals')
        else:
            save_scribble_dir = os.path.join(save_scribble_dir, 'tokens')
            
        os.makedirs(save_scribble_dir, exist_ok=True)
        
        for batch in range(self.batch_size):
            if not save_individual:
                for i, scribble in enumerate(self.scribbles[batch]):
                    phrase_index = self.obj_to_phrase[batch][i]
                    scribble_npy = scribble.cpu().numpy() * 255
                    scribble_img = Image.fromarray(scribble_npy.astype(np.uint8))
                    
                    scribble_path = os.path.join(
                        save_scribble_dir, 
                        "{}.png".format(self.phrases[batch][phrase_index].replace(' ', '_') + '_' + str(self.phrase_indices[batch][phrase_index]) + '_' + str(timestep))
                    )
                    scribble_img.save(scribble_path)
            else:
                for i, scribble in enumerate(self.individual_scribbles[batch]):
                    phrase_index = self.individual_to_phrase[batch][i]
                    scribble_npy = scribble.cpu().numpy() * 255
                    scribble_img = Image.fromarray(scribble_npy.astype(np.uint8))
                    
                    scribble_path = os.path.join(
                        save_scribble_dir, 
                        "{}.png".format(self.phrases[batch][phrase_index].replace(' ', '_') + '_' + str(i) + '_' + str(timestep))
                    )
                    scribble_img.save(scribble_path)
        return
    
    
    def save_masks(self, save_individual=False, save_mask_dir=None, timestep=1000):
        save_mask_dir = save_mask_dir if save_mask_dir is not None else self.save_mask_dirs[0]
        
        if save_individual:
            save_mask_dir = os.path.join(save_mask_dir, 'individuals')
        else:
            save_mask_dir = os.path.join(save_mask_dir, 'tokens')
            
        os.makedirs(save_mask_dir, exist_ok=True)
        
        for batch in range(self.batch_size):
            if not save_individual:
                for i, mask in enumerate(self.masks[batch]):
                    phrase_index = self.obj_to_phrase[batch][i]
                    mask = mask.cpu().numpy() * 255
                    mask = Image.fromarray(mask.astype(np.uint8))
                    
                    mask_path = os.path.join(
                        save_mask_dir, 
                        "{}.png".format(self.phrases[batch][phrase_index].replace(' ', '_') + '_' + str(self.phrase_indices[batch][phrase_index]) + '_' + str(timestep))
                    )
                    mask.save(mask_path)
            else:
                for i, mask in enumerate(self.individual_masks[batch]):
                    phrase_index = self.individual_to_phrase[batch][i]
                    mask = mask.cpu().numpy() * 255
                    mask = Image.fromarray(mask.astype(np.uint8))
                    
                    mask_path = os.path.join(
                        save_mask_dir, 
                        "{}.png".format(self.phrases[batch][phrase_index].replace(' ', '_') + '_' + str(i) + '_' + str(timestep))
                    )
                    mask.save(mask_path)
                
        return
    