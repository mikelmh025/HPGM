from __future__ import print_function
import os
import time
import vutils
import torch
import json
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
from six.moves import range
from miscc.config import cfg
from miscc.utils import mkdir_p, bbox_iou, bbox_refiner
from model_graph import GCN, BBOX_NET, Generator, Discriminator, compute_gradient_penalty
import numpy as np
from torchvision.utils import save_image

plt.switch_backend('agg')


import sys
from torch.autograd import Variable


# ################## Shared functions ###################
def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def define_optimizers(model, lr, weight_decay):
    optimizer_model = optim.Adam(model.parameters(),
                            lr=lr, 
                            weight_decay=weight_decay,
                            betas=(0.5, 0.999))
    return optimizer_model


def save_model(model, epoch, model_dir, model_name, best=False):
    if best:
        torch.save(model.state_dict(), '%s/%s_best.pth' % (model_dir, model_name))
    else:
        torch.save(model.state_dict(), '%s/%s_%d.pth' % (model_dir, model_name, epoch))


def save_img_results(imgs_tcpu, real_box, boxes_pred, count, image_dir):
    num = cfg.TRAIN.VIS_COUNT
    # The range of real_img (i.e., self.imgs_tcpu[i][0:num])
    # is changed to [0, 1] by function vutils.save_image
    real_img = imgs_tcpu[-1][0:num]
    vutils.save_image(
        real_img, '%s/count_%09d_real_samples.png' % (image_dir, count),
        normalize=True)
    # save bounding box images
    vutils.save_bbox(
        real_img, real_box, '%s/count_%09d_real_bbox.png' % (image_dir, count),
        normalize=True)
    # vutils.save_bbox(
    #     real_img, boxes_pred, '%s/count_%09d_fake_bbox.png' % (image_dir, count),
    #     normalize=True)
    # save floor plan images
    vutils.save_floor_plan(
        real_img, real_box, '%s/count_%09d_real_floor_plan.png' % (image_dir, count),
        normalize=True)

    # save_image(boxes_pred, "test.png")
    save_image(boxes_pred.mean(0)*255, '%s/count_%09d_fake_floor_plan.png' % (image_dir, count))
    
    # vutils.save_floor_plan(
    #     real_img, boxes_pred, '%s/count_%09d_fake_floor_plan.png' % (image_dir, count),
    #     normalize=True)


def save_img_results_test(imgs_tcpu, real_box, boxes_pred, count, test_dir):
    num = cfg.TRAIN.VIS_COUNT

    # The range of real_img (i.e., self.imgs_tcpu[i][0:num])
    # is changed to [0, 1] by function vutils.save_image
    real_img = imgs_tcpu[-1][0:num]
    vutils.save_image(
        real_img, '%s/count_%09d_real_samples.png' % (test_dir, count),
        normalize=True)

    # save bounding box images
    vutils.save_bbox(
        real_img, real_box, '%s/count_%09d_real_bbox.png' % (test_dir, count),
        normalize=True)

    vutils.save_bbox(
        real_img, boxes_pred, '%s/count_%09d_fake_bbox.png' % (test_dir, count),
        normalize=True)

    # save floor plan images
    vutils.save_floor_plan(
        real_img, real_box, '%s/count_%09d_real_floor_plan.png' % (test_dir, count),
        normalize=True)

    vutils.save_floor_plan(
        real_img, boxes_pred, '%s/count_%09d_fake_floor_plan.png' % (test_dir, count),
        normalize=True)


def save_img_results_for_FID(imgs_tcpu, fake_imgs, save_path_real, 
                            save_path_fake, step_test, batch_size):
    real_img = imgs_tcpu[-1]
    fake_img = fake_imgs[-1]
    # save image for FID calculation
    vutils.save_image_for_fid(real_img, fake_img, save_path_real, 
                            save_path_fake, step_test, batch_size,
                            normalize=True)


def save_txt_results(text, count, text_dir):
    with open('%s/count_%09d.txt'%(text_dir, count), 'a') as f:
        for t in text:
            f.write('{}\n'.format(t))


def save_txt_results_bbox(boxes, count, text_dir):
    # print(boxes[0][1])
    # assert False
    room_classes = ['livingroom', 'bedroom', 'corridor', 'kitchen', 
                    'washroom', 'study', 'closet', 'storage', 'balcony']
    rooms_counter = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    temp_dir = {}
    temp_dir["rooms"] = {}
    for i in range(len(boxes[0][1])):
        idx = boxes[0][1][i]
        rooms_counter[idx] += 1
        key = "%s%s" % (room_classes[idx], rooms_counter[idx])
        bounding_box = boxes[0][0][i].cpu().detach().numpy()
        temp_dir["rooms"][key] = {"min_x": "%s"%(bounding_box[0]),
                                "min_y": "%s"%(bounding_box[1]),
                                "max_x": "%s"%(bounding_box[2]),
                                "max_y": "%s"%(bounding_box[3])}
    with open('%s/count_%09d.json'%(text_dir, count), 'w') as f:
        json.dump(temp_dir, f)



# ################# Text to image task############################ #
class LayoutTrainer(object):
    def __init__(self, output_dir, dataloader_train, imsize, dataset_test):
        # build save data dir
        self.output_dir = output_dir
        self.model_dir = os.path.join(output_dir, 'Model')
        self.image_dir = os.path.join(output_dir, 'Image')
        self.text_dir = os.path.join(output_dir, 'Text')
        self.log_dir = os.path.join(output_dir, 'Log')
        self.image_dir_test = os.path.join(output_dir, 'Image_test')
        self.text_dir_test = os.path.join(output_dir, 'Text_test')
        self.image_dir_eval = os.path.join(output_dir, 'Image_eval')
        self.text_dir_eval = os.path.join(output_dir, 'Text_eval')
        self.text_dir_eval_gt = os.path.join(output_dir, 'Text_eval_gt')
        self.region_dir_eval = os.path.join(output_dir, 'region_process_eval')
        mkdir_p(self.model_dir)
        mkdir_p(self.image_dir)
        mkdir_p(self.text_dir)
        mkdir_p(self.log_dir)
        mkdir_p(self.image_dir_test)
        mkdir_p(self.text_dir_test)
        mkdir_p(self.image_dir_eval)
        mkdir_p(self.text_dir_eval)
        mkdir_p(self.text_dir_eval_gt)
        mkdir_p(self.region_dir_eval)
        # save the information of cfg
        log_cfg = os.path.join(self.log_dir, 'log_cfg.json')
        with open(log_cfg, 'a') as outfile:
            json.dump(cfg, outfile)
            outfile.write('\n')

        s_gpus = cfg.GPU_ID.split(',')
        self.gpus = [int(ix) for ix in s_gpus]
        self.num_gpus = len(self.gpus)
        self.device = torch.device('cuda:{}'.format(self.gpus[0]) if self.num_gpus>0 else 'cpu')
        cudnn.benchmark = True

        if cfg.TRAIN.FLAG:
            self.batch_size = cfg.TRAIN.BATCH_SIZE * self.num_gpus
        else:
            self.batch_size = cfg.EVAL.BATCH_SIZE
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        if dataloader_train!=None:
            self.dataloader_train = dataloader_train
            self.num_batches = len(self.dataloader_train)
        self.dataloader_test = dataset_test
        self.best_loss = 10000.0
        self.best_epoch = 0

    def prepare_data(self, data):
        label_imgs, _, wrong_label_imgs, _, graph, edge, bbox, rooms_mks, objs_vector, key = data

        vgraph, vbbox, vobjs_vector = [], [], []
        if cfg.CUDA:
            for i in range(len(graph)):
                # vgraph.append((graph[i][0].to(self.device), graph[i][1].to(self.device)))
                vgraph.append(graph[i].to(self.device))
                vbbox.append((bbox[i][0].to(self.device), bbox[i][1].to(self.device)))
                vobjs_vector.append((objs_vector[i][0].to(self.device), objs_vector[i][1].to(self.device)))
        # print(len(rooms_mks))
        # print(len(rooms_mks[0]))
        # print(rooms_mks[0][0].shape)
        rooms_mks = torch.cat(rooms_mks)
        return label_imgs, vgraph, edge, vbbox, rooms_mks, vobjs_vector, key

    def prepare_data_test(self, data):
        # imgs, w_imgs, t_embedding, _ = data
        label_imgs, _, graph, edge, bbox, objs_vector, key = data

        # real_vimgs = []
        vgraph, vbbox, vobjs_vector = [], [], []
        if cfg.CUDA:
            for i in range(len(graph)):
                # vgraph.append((graph[i][0].to(self.device), graph[i][1].to(self.device)))
                vgraph.append(graph[i].to(self.device))
                vbbox.append((bbox[i][0].to(self.device), bbox[i][1].to(self.device)))
                vobjs_vector.append((objs_vector[i][0].to(self.device), objs_vector[i][1].to(self.device)))
        return label_imgs, vgraph, vbbox, vobjs_vector, key

    def define_models(self):
        if cfg.TRAIN.USE_SIZE_AS_INPUT:
            objs_vector_dim = 19
        else:
            objs_vector_dim = 18
        # build gcn model
        input_graph_dim = objs_vector_dim
        hidden_graph_dim = 64
        output_graph_dim = objs_vector_dim
        gcn = GCN(nfeat=input_graph_dim, 
                  nhid=hidden_graph_dim, output_dim=output_graph_dim)
        # build box_net model
        gconv_dim = objs_vector_dim
        gconv_hidden_dim = 512
        box_net_dim = 4
        mlp_normalization = 'none'
        box_net_layers = [gconv_dim, gconv_hidden_dim, box_net_dim]
        box_net = BBOX_NET(box_net_layers, batch_norm=mlp_normalization)
        return gcn, box_net

    def process_other(self, curr, other):
        l_curr = curr.shape[0]
        l_other = other.shape[0]
        if l_curr > l_other:
            # print("inshape : ", curr.shape, other.shape)
            curr = torch.split(curr,[l_other,l_curr-l_other], 0)[0]
        elif l_curr < l_other:
            # print("inshape : ", curr.shape, other.shape)
            other = torch.split(other,[l_curr,l_other-l_curr], 0)[0]
        return curr, other
   
    # Visualize a single batch
    def visualizeSingleBatch(self, data):
        with torch.no_grad():
            # Unpack batch
            imgs_tcpu, graph, edge, real_box, rooms_mks, objs_vector, key = self.prepare_data(data)

            graph_objs_vector_list = []
            objs_vector_list = []
            index_graph_objs_vector = []
            index_graph_edge_vector = []
            for i in range (len(edge)):
                for j in range(edge[i].shape[0]):
                    index_graph_edge_vector.append(j)
            index_graph_edge_vector = torch.LongTensor(index_graph_edge_vector)


            for i in range(len(self.real_box)):
                graph_objs_vector = self.gcn(objs_vector[i][0], graph[i])
                objs_vector_list.append(objs_vector[i][0])          # append x
                graph_objs_vector_list.append(graph_objs_vector)    # append y
                for j in range (0,self.real_box[i][0].shape[0]):
                    index_graph_objs_vector.append(i)
            objs_vector_list_cat = torch.cat(objs_vector_list,dim=0)
            graph_objs_vector_cat = torch.cat(graph_objs_vector_list,dim=0)
            index_graph_objs_vector = torch.LongTensor(index_graph_objs_vector)

            real_mks = Variable(rooms_mks.type(self.Tensor))
            given_nds_x = objs_vector_list_cat   # input nodes X
            given_nds_y = graph_objs_vector_cat  # input node Y
            given_nds = torch.add(given_nds_x, given_nds_y) # input nodes
            given_eds = torch.cat(edge)

            # Generate a batch of images
            z_shape = [objs_vector_list_cat.shape[0], 128]
            z = Variable(self.Tensor(np.random.normal(0, 1, tuple(z_shape))))
            gen_mks = self.generator(z, given_nds, given_eds) #TODO: Add in multi GPU
            
            # Generate image tensors
            real_imgs_tensor = vutils.combine_images_maps(real_mks, given_nds, given_eds, \
                                                index_graph_objs_vector, index_graph_edge_vector)
            fake_imgs_tensor = vutils.combine_images_maps(gen_mks, given_nds, given_eds, \
                                                index_graph_objs_vector, index_graph_edge_vector)

            # Save images
            save_image(real_imgs_tensor, "./exps/{}/{}_real.png".format(exp_folder, batches_done), \
                    nrow=12, normalize=False)
            save_image(fake_imgs_tensor, "./exps/{}/{}_fake.png".format(exp_folder, batches_done), \
                    nrow=12, normalize=False)
        return
    def train(self):
        # plot
        self.training_epoch = []
        self.testing_epoch = []
        self.training_error = []
        self.testing_error = []
        # define models
        if cfg.TRAIN.USE_GCN:
            self.gcn, self.box_net = self.define_models()
        else:
            _, self.box_net = self.define_models()

        ### For GAN ###
        # Loss function
        self.adversarial_loss = torch.nn.BCEWithLogitsLoss()
        
        # Initialize generator and discriminator
        self.generator = Generator()
        self.discriminator = Discriminator()

        # load gcn checkpoints
        if cfg.TRAIN.GCN != '':
            self.gcn.load_state_dict(
                torch.load(cfg.TRAIN.GCN))
        # load G checkpoints
        if cfg.TRAIN.G != '':
            self.generator.load_state_dict(
                torch.load(cfg.TRAIN.G))
        # load D checkpoints
        if cfg.TRAIN.D != '':
            self.discriminator.load_state_dict(
                torch.load(cfg.TRAIN.D))
        # load box_net checkpoints
        if cfg.TRAIN.BOX_NET != '':
            self.box_net.load_state_dict(
                torch.load(cfg.TRAIN.BOX_NET))

        # optimization method
        self.optimizer_gcn = define_optimizers(
            self.gcn, cfg.GCN.LR, cfg.GCN.WEIGHT_DECAY)
        self.optimizer_bbox = define_optimizers(
            self.box_net, cfg.BBOX.LR, cfg.BBOX.WEIGHT_DECAY)
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=0.0001, betas=(0.5, 0.999)) 
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999)) 


        # criterion function
        self.criterion_bbox = nn.MSELoss()

        cuda = True if torch.cuda.is_available() else False
        self.Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        if cfg.CUDA:
            # criterion
            self.criterion_bbox.to(self.device)
            # model
            self.gcn.to(self.device)
            self.box_net.to(self.device)
            self.generator.cuda()
            self.discriminator.cuda()
        predictions = []
        start_epoch = 0
        batches_done = 0
        for epoch in range(start_epoch, self.max_epoch):
            start_t = time.time()
            # ================== #
            #      Training      #
            # ================== #
            for step, data in enumerate(self.dataloader_train, 0):
                #######################################################
                # (0) Prepare training data
                # rooms_mks is the real room masks
                ######################################################
                self.imgs_tcpu, self.graph, self.edge, self.real_box, self.rooms_mks, self.objs_vector, self.key = self.prepare_data(data)
                
                #######################################################
                # (1) Generate layout position
                ######################################################
                self.box_net.train()

                self.real_box_list = []
                self.graph_list = []
                self.objs_vector_list = []
                self.graph_objs_vector_list = []
                self.index_graph_objs_vector = []
                # for each image
                # GCN Graph to Graph
                for i in range(len(self.real_box)):
                    graph_objs_vector = self.gcn(self.objs_vector[i][0], self.graph[i]) # Get one graph
                    #Append real, x, A, y to its lists
                    self.graph_list.append(self.graph[i])
                    self.real_box_list.append(self.real_box[i][0])          # append real
                    self.objs_vector_list.append(self.objs_vector[i][0])    # append x
                    self.graph_objs_vector_list.append(graph_objs_vector)   # append y
                    for j in range (0,self.real_box[i][0].shape[0]):
                        self.index_graph_objs_vector.append(i)

                # concat all the results into one big tensor
                self.edge = torch.cat(self.edge)
                self.real_box_list_cat = torch.cat(self.real_box_list,dim=0)
                self.objs_vector_list_cat = torch.cat(self.objs_vector_list,dim=0)                # x
                self.graph_objs_vector_cat = torch.cat(self.graph_objs_vector_list,dim=0)         # y
                self.index_graph_objs_vector = torch.LongTensor(self.index_graph_objs_vector)

                # Old CNN box
                # boxes_pred = self.box_net(self.objs_vector_list_cat, self.graph_objs_vector_cat)

                # Adversarial ground truths
                self.batch_g = torch.max(self.index_graph_objs_vector) + 1
                self.valid = Variable(self.Tensor(self.batch_g, 1).fill_(1.0), requires_grad=False)
                self.fake = Variable(self.Tensor(self.batch_g, 1).fill_(0.0), requires_grad=False)

                # Configure input
                # self.real_mks = Variable(self.real_box_list_cat.type(self.Tensor))
                self.rooms_mks = Variable(self.rooms_mks.type(self.Tensor))
                given_nds_x = self.objs_vector_list_cat   # input nodes X
                given_nds_y = self.graph_objs_vector_cat  # input node Y
                given_nds = torch.add(given_nds_x, given_nds_y) # input nodes
                given_eds = self.edge                   # input edges W
                nd_to_sample = self.index_graph_objs_vector # index for each nodes

                # Set grads on
                for p in self.discriminator.parameters():
                    p.requires_grad = True

                # ---------------------
                #  Train Discriminator
                # ---------------------
                self.optimizer_D.zero_grad()


                # Generate a batch of images (Fake)
                z_shape = [self.objs_vector_list_cat.shape[0], 128]
                z = Variable(self.Tensor(np.random.normal(0, 1, tuple(z_shape))))
                gen_mks = self.generator(z, given_nds, given_eds) #TODO: Add in multi GPU


W
                # Real images
                real_validity = self.discriminator(self.rooms_mks,given_nds,given_eds,nd_to_sample)

                # Fake images
                fake_validity = self.discriminator(gen_mks.detach(), given_nds.detach(), \
                                          given_eds.detach(), nd_to_sample.detach())

                # Measure discriminator's ability to classify real from generated samples
                gradient_penalty = compute_gradient_penalty(self.discriminator, self.rooms_mks.data, \
                                                        gen_mks.data, given_nds.data, \
                                                        given_eds.data, nd_to_sample.data, \
                                                        None, None)
                lambda_gp = 10
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) \
                        + lambda_gp * gradient_penalty

                n_critic = 10
                # Update discriminator
                if step % n_critic == 0:
                    d_loss.backward(retain_graph=True)
                else:
                    d_loss.backward()
                self.optimizer_D.step()

                # -----------------
                #  Train Generator
                # -----------------
                self.optimizer_gcn.zero_grad()
                self.optimizer_G.zero_grad()
                for p in self.discriminator.parameters():
                    p.requires_grad = False

                # Train the generator every n_critic steps
                
                if step % n_critic == 0:
                    # Generate a batch of images
                    z = Variable(self.Tensor(np.random.normal(0, 1, tuple(z_shape))))
                    gen_mks = self.generator(z, given_nds, given_eds)
                    fake_validity = self.discriminator(gen_mks, given_nds, given_eds, nd_to_sample)
                    # Update generator
                    g_loss = -torch.mean(fake_validity)
                    g_loss.backward()
                    self.optimizer_G.step()
                    self.optimizer_gcn.step()
                    print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                        % (epoch, self.max_epoch, step, len(self.dataloader_train), d_loss.item(), g_loss.item()))
                        
                    if (batches_done % cfg.TRAIN.CHECK_POINT_INTERVAL == 0) and batches_done:
                        save_model(model=self.gcn, epoch=epoch, model_dir=self.model_dir,
                            model_name='gcn', best=False)
                        save_model(model=self.generator, epoch=epoch, model_dir=self.model_dir,
                            model_name='generator', best=False)
                        save_model(model=self.discriminator, epoch=epoch, model_dir=self.model_dir,
                            model_name='discriminator', best=False)
                        save_img_results(self.imgs_tcpu, self.real_box, gen_mks, epoch, self.image_dir)
                        # self.visualizeSingleBatch(data)
                    batches_done += n_critic
                # sys.exit()

                

            # # ================= #
            # #      Valid        #
            # # ================= #
            # if epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:
            #     self.gcn.eval()
            #     self.generator.eval()
            #     self.box_net.eval()
            #     boxes_pred_collection = []
            #     for i in range(len(self.real_box)):
            #         graph_objs_vector = self.gcn(self.objs_vector[i][0], self.graph[i])
            #         # bounding box prediction
            #         boxes_pred_save = self.box_net(self.objs_vector[i][0], graph_objs_vector)
            #         boxes_pred_collection.append((boxes_pred_save, self.real_box[i][1]))
            #     save_img_results(self.imgs_tcpu, self.real_box, boxes_pred_collection, epoch, self.image_dir)
            #     save_txt_results(self.key, epoch, self.text_dir)

            #     # evaluate the model
            #     print('generating the test data...')
            #     for step_test, data_test in enumerate(self.dataloader_test, 0):
            #         # get data
            #         self.imgs_tcpu_test, self.graph_test, self.bbox_test, self.objs_vector_test, self.key_test = self.prepare_data_test(data_test)
            #         boxes_pred_test_collection = []
            #         for i in range(len(self.bbox_test)):
            #             graph_objs_vector_test = self.gcn(self.objs_vector_test[i][0], self.graph_test[i])
            #             # bounding box prediction
            #             boxes_pred_test = self.box_net(self.objs_vector_test[i][0], graph_objs_vector_test)
            #             boxes_pred_test_collection.append((boxes_pred_test, self.bbox_test[i][1]))
            #             # record the loss
            #             if i == 0:
            #                 err_bbox_test = self.criterion_bbox(boxes_pred_test, self.bbox_test[i][0])
            #             else:
            #                 err_bbox_test += self.criterion_bbox(boxes_pred_test, self.bbox_test[i][0])
            #         err_bbox_test = err_bbox_test / len(self.bbox_test)
            #         err_total_test = cfg.TRAIN.COEFF.BBOX_LOSS * err_bbox_test
            #         if step_test == 0:
            #             save_img_results_test(self.imgs_tcpu_test, self.bbox_test, boxes_pred_test_collection, epoch, self.image_dir_test)
            #             save_txt_results(self.key_test, epoch, self.text_dir_test)
            #         break
            #     # plot
            #     self.testing_epoch.append(epoch)
            #     self.testing_error.append(err_total_test)
    #         # ================ #
    #         #      Saving      #
    #         # ================ #
    #         self.training_epoch.append(epoch)
    #         self.training_error.append(err_total)
    #         # plot
    #         plt.figure(0)
    #         plt.plot(self.training_epoch, self.training_error, color="r", linestyle="-", linewidth=1, label="training")
    #         plt.plot(self.testing_epoch, self.testing_error, color="b", linestyle="-", linewidth=1, label="testing")
    #         plt.xlabel("epoch")
    #         plt.ylabel("loss")
    #         plt.legend(loc='best')
    #         plt.savefig(os.path.join(self.output_dir, "loss.png"))
    #         plt.close(0)

    #         # loss
    #         with open(os.path.join(self.log_dir, 'log_loss.txt'), 'a') as f:
    #             f.write('{},{}\n'.format(epoch, self.training_error[-1]))
    #         # print
    #         end_t = time.time()
    #         try:
    #             print('[%d/%d][%d] Loss_total: %.5f Loss_bbox: %.5f Time: %.2fs' % (epoch, self.max_epoch,
    #                   self.num_batches, 0, cfg.TRAIN.COEFF.BBOX_LOSS*err_bbox, end_t - start_t))
    #         except IOError as e:
    #             print(e)
    #             pass
    #         # for cfg.TRAIN.CHECK_POINT_INTERVAL times save one model
    #         print('saving checkpoint models...')
    #         if epoch % cfg.TRAIN.CHECK_POINT_INTERVAL == 0:
    #             save_model(model=self.gcn, epoch=epoch, model_dir=self.model_dir,
    #                        model_name='gcn', best=False)
    #             save_model(model=self.box_net, epoch=epoch, model_dir=self.model_dir,
    #                        model_name='box_net', best=False)

    # # evaluate the trained models
    # def evaluate(self):
    #     # define models
    #     self.gcn, self.box_net = self.define_models()
    #     # load gcn
    #     if cfg.EVAL.GCN != '':
    #         self.gcn.load_state_dict(
    #             torch.load(os.path.join(cfg.EVAL.OUTPUT_DIR, 'Model',
    #                        cfg.EVAL.GCN)))
    #     # load box_net
    #     if cfg.EVAL.BOX_NET != '':
    #         self.box_net.load_state_dict(
    #             torch.load(os.path.join(cfg.EVAL.OUTPUT_DIR, 'Model',
    #                        cfg.EVAL.BOX_NET)))
    #     if cfg.CUDA:
    #         self.gcn.to(self.device)
    #         self.box_net.to(self.device)
    #     self.gcn.eval()
    #     self.box_net.eval()

    #     # evaluate the model
    #     print('evaluating the test data...')
    #     total_IoU = 0.0
    #     count_boxes = 0
    #     for step_test, data_test in enumerate(self.dataloader_test, 1):
    #         # get data
    #         self.imgs_tcpu_test, self.graph_test, self.bbox_test, self.objs_vector_test, self.key_test = self.prepare_data_test(data_test)
    #         boxes_pred_test_collection = []
    #         boxes_pred_test_collection_gt = []
    #         for i in range(len(self.bbox_test)):
    #             graph_objs_vector_test = self.gcn(self.objs_vector_test[i][0], self.graph_test[i])
    #             # bounding box prediction
    #             boxes_pred_test = self.box_net(self.objs_vector_test[i][0], graph_objs_vector_test)
    #             IoU, num_boxes = bbox_iou(boxes_pred_test, self.bbox_test[i][0])
    #             total_IoU += IoU
    #             count_boxes += num_boxes
    #             boxes_pred_test_collection.append((boxes_pred_test, self.bbox_test[i][1]))
    #             boxes_pred_test_collection_gt.append((self.bbox_test[i][0], self.bbox_test[i][1]))
    #         # save layout images and texts
    #         save_img_results_test(self.imgs_tcpu_test, self.bbox_test, boxes_pred_test_collection, step_test, self.image_dir_eval)
    #         save_txt_results_bbox(boxes_pred_test_collection, step_test, self.text_dir_eval)
    #         save_txt_results_bbox(boxes_pred_test_collection_gt, step_test, self.text_dir_eval_gt)
    #         # region Processing
    #         from regionProcessing import RegionProcessor,get_merge_image
    #         room_classes = ['livingroom', 'bedroom', 'corridor', 'kitchen',
    #                         'washroom', 'study', 'closet', 'storage', 'balcony']
    #         coord_data = [boxes_pred_test.cpu(), self.bbox_test[0][1], room_classes]
    #         processor = RegionProcessor(coord_data=coord_data)
    #         lines, rooms = processor.get_lines_from_json()
    #         print(lines, rooms)
    #         get_merge_image(lines, rooms, processor, self.region_dir_eval, step_test)
    #     print('Avg IoU: {}'.format(total_IoU/count_boxes))
