import torch
import math
irange = range

from miscc.config import cfg
import os
import cv2
import numpy as np

def make_grid(tensor, nrow=8, padding=2,
              normalize=False, range=None, 
              scale_each=False, pad_value=0):
    """Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The Final grid size is (B / nrow, nrow). Default is 8.
        padding (int, optional): amount of padding. Default is 2.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by subtracting the minimum and dividing by the maximum pixel value.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If True, scale each image in the batch of
            images separately rather than the (min, max) over all images.
        pad_value (float, optional): Value for the padded pixels.

    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_

    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), "range has to be a tuple (min, max) if specified. min and max are numbers"

        # def norm_ip(img, min, max):
        #     img.clamp_(min=min, max=max)
        #     img.add_(-min).div_(max - min + 1e-5)

        # def norm_range(t, range):
        #     if range is not None:
        #         norm_ip(t, range[0], range[1])
        #     else:
        #         # print('min', float(t.min()))
        #         # print('max', float(t.max()))
        #         # assert False
        #         norm_ip(t, float(t.min()), float(t.max()))

        def norm_ip(img):
            img.mul_(0.5).add_(0.5)

        def norm_range(t, range):
            norm_ip(t)

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)


    if tensor.size(0) == 1:
        return tensor.squeeze()

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    grid = tensor.new_full((3, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding)\
                .narrow(2, x * width + padding, width - padding)\
                .copy_(tensor[k])
            k = k + 1
    return grid


def make_grid_bbox(tensor, box, nrow=8, padding=2,
              normalize=False, range=None, 
              scale_each=False, pad_value=0):
    """Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The Final grid size is (B / nrow, nrow). Default is 8.
        padding (int, optional): amount of padding. Default is 2.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by subtracting the minimum and dividing by the maximum pixel value.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If True, scale each image in the batch of
            images separately rather than the (min, max) over all images.
        pad_value (float, optional): Value for the padded pixels.

    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_

    """

    # make the mini-batch of images into a grid
    # nmaps = tensor.size(0)
    nmaps = len(box)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    # height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    height, width = int(256 + padding), int(256 + padding)
    tensor = torch.ones(())
    grid = tensor.new_full((3, height * ymaps + padding, width * xmaps + padding), pad_value)
    # # add the white image into the grid
    # block = tensor.new_full((3, height - padding, width - padding), 9.0/13)
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            # add the white image into the grid
            block = tensor.new_full((3, height - padding, width - padding), 9.0/13)
            # print(box[0].size())
            # print(box[1].size())
            # assert False
            # num_curr_box = box[0][k].size(0)
            num_curr_box = box[k][0].size(0)
            for z in irange(num_curr_box):
                # label = box[1][k][z].item()
                try:
                    label = box[k][1][z].item()
                except:
                    print(box)
                    print(k)
                    assert False
                
                if label!=-1:
                    block = draw_box(block, box[k][0][z], label)
                    # print(k, z)
                else:
                    break
            # copy to the grid
            grid.narrow(1, y * height + padding, height - padding)\
                .narrow(2, x * width + padding, width - padding)\
                .copy_(block)
            k = k + 1
    return grid


def draw_box(image, curr_box, label):
    """Draw 3-pixel width bounding boxes on the given image array.
    color: list of 3 int values for RGB.
    """
    # y1, x1, y2, x2 = box
    # print(curr_box)
    # assert False
    x1, y1, x2, y2 = curr_box[0], curr_box[1], curr_box[2], curr_box[3]
    _, h, w = image.size()
    x1 = int(x1.item() * w)
    y1 = int(y1.item() * h)
    x2 = int(x2.item() * w)
    y2 = int(y2.item() * h)
    image[:, y1:y1 + 3, x1:x2] = label/13.0
    image[:, y2:y2 + 3, x1:x2] = label/13.0
    image[:, y1:y2, x1:x1 + 3] = label/13.0
    image[:, y1:y2, x2:x2 + 3] = label/13.0
    return image


def make_grid_floor_plan(tensor, box, nrow=8, padding=2,
              normalize=False, range=None, 
              scale_each=False, pad_value=0):
    """Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The Final grid size is (B / nrow, nrow). Default is 8.
        padding (int, optional): amount of padding. Default is 2.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by subtracting the minimum and dividing by the maximum pixel value.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If True, scale each image in the batch of
            images separately rather than the (min, max) over all images.
        pad_value (float, optional): Value for the padded pixels.

    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_

    """
    # make the mini-batch of images into a grid
    # nmaps = tensor.size(0)
    nmaps = len(box)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    # height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    height, width = int(256 + padding), int(256 + padding)
    tensor = torch.ones(())
    grid = tensor.new_full((3, height * ymaps + padding, width * xmaps + padding), pad_value)
    # # add the white image into the grid
    # block = tensor.new_full((3, height - padding, width - padding), 9.0/13)

    wall_thickness = 2
    wall_symbol = 2.0

    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            # add the white image into the grid
            block = tensor.new_full((3, height - padding, width - padding), 9.0/13)
            num_curr_box = box[k][0].size(0)
            
            # sorted the box according to their size
            sorted_box = {}
            for z in irange(num_curr_box):
                curr_box = box[k][0][z]
                x1, y1, x2, y2 = curr_box[0], curr_box[1], curr_box[2], curr_box[3]
                sorted_box[z] = (x2-x1)*(y2-y1)
            # to get sorted id
            sorted_box = sorted(sorted_box.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)

            # obtain the sorted box and corresponding label
            for m in irange(num_curr_box):
                # get sorted id
                z = sorted_box[m][0]
                # label = box[1][k][z].item()
                try:
                    label = box[k][1][z].item()
                except:
                    print(box)
                    print(k)
                    assert False
                # draw box in the current image
                if label!=-1:
                    block = draw_floor_plan(block, box[k][0][z], label)
                    # print(k, z)
                else:
                    break

            # copy the current image to the grid
            grid.narrow(1, y * height + padding, height - padding)\
                .narrow(2, x * width + padding, width - padding)\
                .copy_(block)
            k = k + 1
    return grid

def draw_floor_plan(image, curr_box, label):
    """Draw 3-pixel width bounding boxes on the given image array.
    color: list of 3 int values for RGB.
    """
    wall_thickness = 2
    wall_symbol = 2.0
    x1, y1, x2, y2 = curr_box[0], curr_box[1], curr_box[2], curr_box[3]
    _, h, w = image.size()
    x1 = int(x1.item() * w)
    y1 = int(y1.item() * h)
    x2 = int(x2.item() * w)
    y2 = int(y2.item() * h)
    image[:, y1:y2, x1:x2] = label/13.0
    image[:, y1-wall_thickness:y1+wall_thickness, x1:x2] = wall_symbol
    image[:, y2-wall_thickness:y2+wall_thickness, x1:x2] = wall_symbol
    image[:, y1:y2, x1-wall_thickness:x1+wall_thickness] = wall_symbol
    image[:, y1:y2, x2-wall_thickness:x2+wall_thickness] = wall_symbol
    return image


def save_image(tensor, filename, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)

    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    # ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    ndarr = grid.mul_(13).add_(0.5).clamp_(0, 13).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

    # colorize the gray images
    # ndarr = cv2.applyColorMap(cv2.convertScaleAbs(ndarr, alpha=20.0), cv2.COLORMAP_JET)
    palette=[]
    for i in xrange(256):
        palette.extend((255,255,255))
    palette[:3*14]=np.array([[84, 139, 84],
                            [0, 100, 0],
                            [0, 0, 128],
                            [85, 26, 139],
                            [255, 0, 255],
                            [165, 42, 42],
                            [139, 134, 130],
                            [205, 198, 115],
                            [139, 58, 58],
                            [255, 255, 255],
                            [0, 0, 0],
                            [30, 144, 255],
                            [135, 206, 235],
                            [255, 255, 0]], dtype='uint8').flatten()

    im = Image.fromarray(ndarr).convert('L')
    # colorize
    im.putpalette(palette)
    im.save(filename)


def save_bbox(tensor, box, filename, nrow=8, padding=0,
               normalize=False, range=None, scale_each=False, pad_value=0):
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image
    
    # print(box)
    # assert False

    grid = make_grid_bbox(tensor, box, nrow=nrow, padding=padding, pad_value=pad_value,
                        normalize=normalize, range=range, scale_each=scale_each)

    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    # ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    ndarr = grid.mul_(13).add_(0.5).clamp_(0, 13).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

    # colorize the gray images
    # ndarr = cv2.applyColorMap(cv2.convertScaleAbs(ndarr, alpha=20.0), cv2.COLORMAP_JET)
    palette=[]
    for i in xrange(256):
        palette.extend((255,255,255))
    palette[:3*14]=np.array([[84, 139, 84],
                            [0, 100, 0],
                            [0, 0, 128],
                            [85, 26, 139],
                            [255, 0, 255],
                            [165, 42, 42],
                            [139, 134, 130],
                            [205, 198, 115],
                            [139, 58, 58],
                            [255, 255, 255],
                            [0, 0, 0],
                            [30, 144, 255],
                            [135, 206, 235],
                            [255, 255, 0]], dtype='uint8').flatten()

    # # draw box
    # ndarr = draw_box(ndarr, box, palette)

    im = Image.fromarray(ndarr).convert('L')
    # colorize
    im.putpalette(palette)
    im.save(filename)


def save_floor_plan(tensor, box, filename, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image
    
    # print(box)
    # assert False

    grid = make_grid_floor_plan(tensor, box, nrow=nrow, padding=padding, pad_value=pad_value,
                            normalize=normalize, range=range, scale_each=scale_each)

    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    # ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    # ndarr = grid.mul_(13).add_(0.5).clamp_(0, 13).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    ndarr = grid.mul_(13).add_(0.5).clamp_(0, 14).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

    # colorize the gray images
    # ndarr = cv2.applyColorMap(cv2.convertScaleAbs(ndarr, alpha=20.0), cv2.COLORMAP_JET)
    palette=[]
    for i in xrange(256):
        palette.extend((255,255,255))
    palette[:3*15]=np.array([[84, 139, 84],
                            [0, 100, 0],
                            [0, 0, 128],
                            [85, 26, 139],
                            [255, 0, 255],
                            [165, 42, 42],
                            [139, 134, 130],
                            [205, 198, 115],
                            [139, 58, 58],
                            [255, 255, 255],
                            [0, 0, 0],
                            [30, 144, 255],
                            [135, 206, 235],
                            [255, 255, 0],
                            [0, 0, 0]], dtype='uint8').flatten()

    # # draw box
    # ndarr = draw_box(ndarr, box, palette)

    im = Image.fromarray(ndarr).convert('L')
    # colorize
    im.putpalette(palette)
    im.save(filename)



def save_image_for_fid(tensor_real, tensor_fake, save_path_real, 
                save_path_fake, step_test, batch_size,
                nrow=1, padding=0, normalize=False, 
                range=None, scale_each=False, pad_value=0):
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image

    num_imgs = tensor_real.size(0)
    for i in xrange(num_imgs):
        # save real images
        grid_real = make_grid(tensor_real[i], nrow=nrow, padding=padding, pad_value=pad_value,
                         normalize=normalize, range=range, scale_each=scale_each)
        # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        # ndarr_real = grid_real.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        ndarr_real = grid_real.mul_(13).add_(0.5).clamp_(0, 13).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        im_real = Image.fromarray(ndarr_real)
        filename_real = os.path.join(save_path_real, \
            '{:0>4}.png'.format(step_test*batch_size+i))
        im_real.save(filename_real)

        # save fake images
        grid_fake = make_grid(tensor_fake[i], nrow=nrow, padding=padding, pad_value=pad_value,
                         normalize=normalize, range=range, scale_each=scale_each)
        # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        # ndarr_fake = grid_fake.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        ndarr_fake = grid_fake.mul_(13).add_(0.5).clamp_(0, 13).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        im_fake = Image.fromarray(ndarr_fake)
        filename_fake = os.path.join(save_path_fake, \
            '{:0>4}.png'.format(step_test*batch_size+i))
        im_fake.save(filename_fake)




import webcolors
from PIL import Image, ImageDraw, ImageOps, ImageFilter, ImageFont, ImageColor

ID_COLOR = {1: 'brown', 2: 'magenta', 3: 'orange', 4: 'gray', 5: 'red', 6: 'blue', 7: 'cyan', 8: 'green', 9: 'salmon', 10: 'yellow'}

def combine_images_maps(maps_batch, nodes_batch, edges_batch, \
                        nd_to_sample, ed_to_sample, im_size=256):
    maps_batch = maps_batch.detach().cpu().numpy()
    nodes_batch = nodes_batch.detach().cpu().numpy()
    edges_batch = edges_batch.detach().cpu().numpy()
    batch_size = torch.max(nd_to_sample) + 1
    
    all_imgs = []
    shift = 0
    for b in range(batch_size):
        inds_nd = np.where(nd_to_sample==b)
        inds_ed = np.where(ed_to_sample==b)
        
        mks = maps_batch[inds_nd]
        nds = nodes_batch[inds_nd]
        eds = edges_batch[inds_ed]
        
        comb_img = np.ones((im_size, im_size, 3)) * 255
        extracted_rooms = []
        for mk, nd in zip(mks, nds):
            r =  im_size/mk.shape[-1]
            x0, y0, x1, y1 = np.array(mask_to_bb(mk)) * r 
            h = x1-x0
            w = y1-y0
            if h > 0 and w > 0:
                extracted_rooms.append([mk, (x0, y0, x1, y1), nd])
        
        # # draw graph
        # graph_img = draw_graph(nds, eds, shift, im_size=im_size)
        # shift += len(nds)
        # all_imgs.append()graph_img
        
        # draw masks
        mask_img = np.ones((32, 32, 3)) * 255
        for rm in extracted_rooms:
            mk, _, nd = rm 
            inds = np.array(np.where(mk>0))
            _type = np.where(nd==1)[0]
            if len(_type) > 0:
                color = ID_COLOR[_type[0] + 1]
            else:
                color = 'black'
            r, g, b = webcolors.name_to_rgb(color)
            mask_img[inds[0, :], inds[1, :], :] = [r, g, b]
        mask_img = Image.fromarray(mask_img.astype('uint8'))
        mask_img = mask_img.resize((im_size, im_size))
        all_imgs.append(torch.FloatTensor(np.array(mask_img).transpose(2, 0, 1))/255.0)
            
        # draw boxes - filling
        comb_img = Image.fromarray(comb_img.astype('uint8'))
        dr = ImageDraw.Draw(comb_img)
        for rm in extracted_rooms:
            _, rec, nd = rm 
            dr.rectangle(tuple(rec), fill='beige')
            
        # draw boxes - outline
        for rm in extracted_rooms:
            _, rec, nd = rm 
            _type = np.where(nd==1)[0]
            if len(_type) > 0:
                color = ID_COLOR[_type[0] + 1]
            else:
                color = 'black'
            dr.rectangle(tuple(rec), outline=color, width=4)
            
#         comb_img = comb_img.resize((im_size, im_size))
        all_imgs.append(torch.FloatTensor(np.array(comb_img).\
                                     astype('float').\
                                     transpose(2, 0, 1))/255.0)
    all_imgs = torch.stack(all_imgs)
    return all_imgs


def draw_graph(nds, eds, shift, im_size=128):

    # Create graph
    graph = AGraph(strict=False, directed=False)

    # Create nodes
    for k in range(nds.shape[0]):
        nd = np.where(nds[k]==1)[0]
        if len(nd) > 0:
            color = ID_COLOR[nd[0]+1]
            name = '' #CLASS_ROM[nd+1]
            graph.add_node(k, label=name, color=color)

    # Create edges
    for i, p, j in eds:
        if p > 0:
            graph.add_edge(i-shift, j-shift, color='black', penwidth='4')
    
    graph.node_attr['style']='filled'
    graph.layout(prog='dot')
    graph.draw('temp/_temp_{}.png'.format(EXP_ID))

    # Get array
    png_arr = open_png('temp/_temp_{}.png'.format(EXP_ID), im_size=im_size) 
    im_graph_tensor = torch.FloatTensor(png_arr.transpose(2, 0, 1)/255.0)
    return im_graph_tensor

def mask_to_bb(mask):
    
    # get masks pixels
    inds = np.array(np.where(mask>0))
    
    if inds.shape[-1] == 0:
        return [0, 0, 0, 0]

    # Compute BBs
    y0, x0 = np.min(inds, -1)
    y1, x1 = np.max(inds, -1)

    y0, x0 = max(y0, 0), max(x0, 0)
    y1, x1 = min(y1, 255), min(x1, 255)

    w = x1 - x0
    h = y1 - y0
    x, y = x0, y0
    
    return [x0, y0, x1+1, y1+1]