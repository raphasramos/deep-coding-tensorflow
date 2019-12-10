""" Implementation of image autoencoder and its dynamics. """


import tensorflow as tf
from pathlib import Path
from itertools import repeat, chain
from PIL import Image
import glob
from skimage.util import view_as_blocks
from copy import deepcopy
from img_common.enums import *
import sys

CURSOR_UP_ONE = '\x1b[1A'
ERASE_LINE = '\x1b[2K'

class AutoEnc:
    """
    Class representing the image autoencoders. It has all methods necessary
    to operate with them.
    """
    def __init__(self, json_configs):
        self.m_cfg = json_configs['model'].copy()
        self.r_cfg = json_configs['run'].copy()

        self.step = 0
        self.img_paths = None
        self.opt = []
        self.vars = []
        self.loss = None
        self.patches_ref = 0
        self.curr_img = None
        self.count_patches = 0

        self.custom_objects = {'ConvBin': ConvBin,
                               'DepthToSpace': DepthToSpace,
                               'CustomConv2DLSTM': CustomConv2DLSTM}

        # TODO: make it work with different levels (like, training another
        #  version of the network)
        self.levels = self.m_cfg['configs']['levels']
        self._load_model() if self.m_cfg['load_model'] else self._create_model()

        self.curr_patches = [[] for _ in range(self.levels)]
        self.curr_latents = [[] for _ in range(self.levels)]

        out = Path(self.r_cfg['out_folder'])
        cont = 0
        while out.exists():
            out = Path(self.r_cfg['out_folder']).with_suffix('.' + str(cont))
            cont += 1
        out.mkdir(parents=True, exist_ok=True)
        self.out_folder = out

        self.ckpt_iter = [-1]
        if self.r_cfg['train']:
            num_iter = self.r_cfg['data_loader']['num_data']
            num_digits = str(len(str(num_iter)))
            self.ckpt_fmt = str((self.out_folder / str(Folders.CHECKPOINTS)) /
                                ('{:0' + num_digits + '.0f}'))
            self.ckpt_iter = list(np.unique(np.ceil(np.linspace(
                0, num_iter - 1, self.r_cfg['num_ckpt']))).astype(int))

    def _set_level_ops(self, input_t, level, layers=None):
        """
        Instantiate series of operations that correspond to one level
        of iteration. Already instantiated layers can be passed as argument.

        @type input_t: tf.Tensor
        @param input_t: Input tensor shape the model accepts. Shape has the
                        information of batch size, height, width and number of
                        channels.
        @type level: int
        @param level: Number of residues to be considered.
        @type layers: list
        @param layers: Keras layers already instantiated (recursive).
        @return: Returns two tensors and a list.
                 1. The latent tensor corresponds to the shape of the latent
                 defined by the binarizer.
                 2. The output tensor corresponds to the shape of the output
                    (must be the same of the input)
                 3. The list contains two elements: the encoder and decoder
                    Keras instantiated layers (using Enums).
                    These layers are defined in the json file.
        """
        tensor, output_t = input_t, []
        if layers:
            for layer_group in layers:
                for layer in layer_group:
                    tensor = layer(tensor)
                output_t.append(tensor)
        else:
            layers = [[], []]
            enc_spec = list(map(deepcopy, self.m_cfg['new_model']['enc']))
            dec_spec = list(map(deepcopy, self.m_cfg['new_model']['dec']))
            for cont, spec in enumerate([enc_spec, dec_spec]):
                for layer_cfg in spec:
                    layer_cfg['name'] = str(level) + '/' + layer_cfg['name']
                    layer = KLayers(layer_cfg.pop('type')).value(**layer_cfg)
                    tensor = layer(tensor)
                    layers[cont] += [layer]
                output_t.append(tensor)

        latent_t = output_t[0]
        output_t = output_t[1]
        return latent_t, output_t, layers

    def _inst_optimizer(self):
        """
        Function that creates a optimizer.
        @rtype:  tensorflow.pythone.keras.optimizer_v2.adam.Adam
        @return: Returns the optimizer object instantiated. The learning rate
                 will be as defined in the json file.
        """
        optimizer = Optimizers(self.m_cfg['configs']['lr_politics']['optimizer']).value
        lr_schedule = self.m_cfg['configs']['lr_politics']['lr']
        opt = optimizer(learning_rate=lr_schedule)
        return opt

    def _load_model(self):
        """
        Function to read a loaded model and prepare useful variables.
        It instantiates one optimizer for each one of the models (levels
        of residues). The loss is the same for all the models.
        """
        self.model = tf.keras.experimental.load_from_saved_model(
            self.m_cfg['load_model'], custom_objects=self.custom_objects)

        ref = 1 if self.m_cfg['configs']['recursive'] else self.levels
        self.opt = [self._inst_optimizer() for _ in range(ref)]
        self.loss = Losses(self.m_cfg['configs']['loss']).value

        l_groups = np.split(np.array(self.model.layers), ref)
        self.vars = list(map(
            lambda g: list(chain(*map(lambda e: e.variables, g))), l_groups))

    def _create_model(self):
        """
        This method is responsible for prototyping the keras model. Basically,
        it does three things:
        1. Instantiate the layers defined in the json file by calling
           L{_set_level_ops}.

        2. Instantiate the loss and the optimizer defined.

        3. Instantiate the Keras model (one for each level).
        """
        ref = 0 if self.m_cfg['configs']['recursive'] else -1
        out_t, l_t, models = [], [], []
        in_t = [tf.keras.Input(batch_size=self.m_cfg['configs']['batch'],
                               shape=self.m_cfg['configs']['patch'])]
        for level in np.arange(self.levels):
            if not self.m_cfg['configs']['recursive'] or not level:
                lat, res, layers = self._set_level_ops(in_t[-1], level)
                opt = self._inst_optimizer()
                self.opt += [opt]
                curr_layers = sum(layers, [])
                vars = sum(list(map(lambda l: l.variables, curr_layers)), [])
                self.vars.append(vars)
            elif self.m_cfg['configs']['recursive']:
                lat, res, layers = self._set_level_ops(in_t[-1], level, layers)

            out_t += [res]
            l_t += [lat]
            in_t += [tf.keras.layers.Subtract()([in_t[ref], out_t[-1]])]

        inputs, outputs = in_t[0], [in_t[:-1], l_t, out_t]
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.loss = Losses(self.m_cfg['configs']['loss']).value

    def _log_vars_in_tsb(self, inputs, outputs, latents, losses, grads):
        """
        This method logs all useful variables in tensorboard when the execution
        mode is training. This method is called for each batch.

        @type inputs: list
        @param inputs: tf.Tensor corresponding to the input
        @type outputs: list
        @param outputs: tf.Tensor corresponding to model outputs
        @type latents: list
        @param latents: tf.Tensor corresponding to model latents
        @type losses: list
        @param losses: tf.Tensor corresponding to model loss
        @type grads: list
        @param grads: tf.Tensor corresponding to model gradients
        """
        levels = np.arange(self.levels).astype(str)
        conc = lambda *x: np.core.defchararray.add(*x)
        hist = lambda n, v: tf.summary.histogram(n, v, step=self.step)
        scalar = lambda n, v: tf.summary.scalar(n, v, step=self.step)

        list(map(lambda var: hist(var.name, var), self.model.variables))
        list(map(lambda g: list(map(hist, conc('grads', levels), g)), grads))
        list(map(hist, conc('input', levels), inputs))
        list(map(hist, conc('output', levels), outputs))
        list(map(hist, conc('latent', levels), latents))
        list(map(scalar, conc('loss', levels), losses))

    def _save_data(self, outputs, latents):
        """
        Function that saves model outputs and latents. This function uses
        L{calc_n_patches} to know the number of patches that each image
        generates. self.count_patches it's incremented for each batch and
        represents the number of patches corresponding to the current image.
        self.patches_ref it's a constant representing the number of patches
        that the current image generates. When the while condition is True
        (enough patches to reconstruct the image), the image is
        reconstructed from each one of the reconstructed patches
        (model output) in self.curr_patches and is saved to disk. Also,
        self.patches_ref it's updated to the next image.

        @type outputs: list
        @param outputs: Each element is a tf.Tensor representing the
                        reconstructed patches. The shape is (N, H, W, C), where
                        N is the number of patches (maximum value is batch size)
                        , H is the Height, W is the Width and C is the number of
                        channels.
        @type latents: list
        @param latents: Each element is a tf.Tensor representing the latents
                        generated of each patch. The total number of distinct
                        elements from the tf.Tensor is 2 (binarized elements).
        """
        list(map(lambda e, v: e.append(v), self.curr_patches, outputs))
        list(map(lambda e, v: e.append(v), self.curr_latents, latents))
        self.count_patches += len(self.curr_patches[0][-1])

        while self.count_patches >= self.patches_ref:
            aux_patches = list(map(lambda e: tf.concat(e, axis=0),
                                   self.curr_patches))
            aux_latents = list(map(lambda e: tf.concat(e, axis=0),
                                   self.curr_latents))

            data_name = str((self.out_folder / str(Folders.RAW_DATA))
                            / self.curr_img.stem)
            for lvl, (aux_p, aux_l) in enumerate(zip(aux_patches, aux_latents)):
                patches, remaining = tf.split(
                    aux_p, [self.patches_ref, len(aux_p) - self.patches_ref],
                    axis=0)
                self.curr_patches[lvl] = [remaining]
                latent, remaining = tf.split(
                    aux_l, [self.patches_ref, len(aux_l) - self.patches_ref],
                    axis=0)
                self.curr_latents[lvl] = [remaining]

                patch_size = len(patches[0])
                with Image.open(self.curr_img) as img:
                    # width, height -> height, width (to be compatible with
                    # arrays)
                    img_size = np.array(img.size)[::-1]
                p_size = np.ceil(img_size / patch_size).astype(int)

                img = tf.reshape(
                    patches, [*p_size, *self.m_cfg['configs']['patch']])
                img = tf.transpose(img, [0, 2, 1, 3, 4])
                img = tf.reshape(img, [*patch_size * p_size, 3])
                img = tf.image.convert_image_dtype(img, tf.uint8)
                img = img[:img_size[0], :img_size[1], :]

                latent_name = data_name + '_latent' + str(lvl)
                serialized = tf.io.serialize_tensor(latent)
                tf.io.write_file(latent_name, serialized)

                img_name = data_name + '_' + str(lvl) + self.curr_img.suffix
                tf.io.write_file(img_name, tf.image.encode_png(
                    img, compression=0))

            with open(Path(data_name).with_suffix('.txt'), 'w') as file:
                file.write(str(self.curr_img))

            self.count_patches = len(self.curr_patches[0][0])
            if len(self.img_paths):
                self.curr_img = Path(self.img_paths.pop(0))
                self.patches_ref = self.calc_n_patches(
                    self.curr_img, self.m_cfg['configs']['patch'][0])

    @staticmethod
    def calc_n_patches(img_ref, patch_size):
        """
        Calculates the number of patches images with this height and
        width considering the padding.

        @type img_ref: PosixPath
        @param img_ref: Path of the image to be considered.
        @type patch_size: number
        @param patch_size: Number representing the height and width of the
                           patch.

        @rtype: number
        @return: number of patch_size patches that img_ref generates.
        """
        with Image.open(img_ref) as img:
            width, height = img.size

        line_patches = np.ceil(height / patch_size).astype(int)
        column_patches = np.ceil(width / patch_size).astype(int)
        num_of_patches = line_patches * column_patches
        return num_of_patches

    @staticmethod
    @tf.function
    def _load_img(filename):
        """
        Auxiliary loader function to read valid images

        @rtype: tf.Tensor
        @return: img converted to tf.float32 dtype
        """
        img_str = tf.io.read_file(filename)
        img_decoded = tf.image.decode_image(img_str)
        img = tf.image.convert_image_dtype(img_decoded, tf.float32)
        return img

    @staticmethod
    @tf.function
    def _extract_patches(img, patch_s):
        """
        Auxiliary loader function to extract patches from images

        @type img: tf.Tensor
        @param img: Tensor representing the image

        @rtype: np.array
        @return: array containing the patches extracted from img
        """
        def np_extract_patches(img):
            orig = np.array(img.shape[:2])
            new = patch_s[0] * np.ceil(orig / patch_s[0]).astype(int)
            points = new - orig
            img = np.pad(img, [(0, points[0]), (0, points[1]), (0, 0)],
                         mode='constant')
            patches = view_as_blocks(img, tuple(patch_s)).astype(np.float32)
            patches = patches.reshape(-1, *patch_s)
            return patches

        patches = tf.numpy_function(np_extract_patches, [img], tf.float32)
        return patches

    def create_db_loder(self, img_buffer=5000, patch_buffer=20000):
        """
        Create a tensorflow database loader that will load batches from disk.

        @rtype: Dataset (Tensorflow object)
        @return: the iterable object containing the batches of patches.
        """
        files = sorted(glob.glob(self.r_cfg['data_loader']['recursive_glob'],
                                 recursive=True))
        if not self.r_cfg['train'] and self.r_cfg['data_loader']['num_data']:
            files = files[:self.r_cfg['data_loader']['num_data']]
        self.img_paths = files
        if not len(self.img_paths):
            raise ValueError('No image matched the glob!')

        db = tf.data.Dataset.from_tensor_slices(self.img_paths)
        if self.r_cfg['train']:
            db = db.repeat()
            db = db.shuffle(buffer_size=img_buffer)
        db = db.map(self._load_img, num_parallel_calls=4)
        db = db.map(lambda i: self._extract_patches(i, self.m_cfg['configs']['patch']),
                    num_parallel_calls=4)
        db = db.apply(tf.data.experimental.unbatch())
        if self.r_cfg['train']:
            db = db.shuffle(buffer_size=patch_buffer)
        db = db.batch(self.m_cfg['configs']['batch'])
        db = db.prefetch(1)
        if self.r_cfg['train']:
            db = db.take(self.r_cfg['data_loader']['num_data'])

        return db

    @staticmethod
    def _clear_last_lines(n=1):
        """ Clear the last n lines in stdout """
        for _ in range(n):
            sys.stdout.write(CURSOR_UP_ONE)
            sys.stdout.write(ERASE_LINE)

    def run(self, summ_period=50):
        """
        Function that executes the model based on the parameters passed.
        If the execution mode is 'train' the model is saved to disk.
        Otherwise outputs generated by the model are saved to disk.

        @type summ_period: int
        @param summ_period: number of steps for each log in tensorboard
        """
        new_min = self.m_cfg['configs']['pixels_range']['in'][0]
        new_max = self.m_cfg['configs']['pixels_range']['in'][1]
        old_min = self.m_cfg['configs']['pixels_range']['out'][0]
        old_max = self.m_cfg['configs']['pixels_range']['out'][1]
        p_transf = lambda t, old_min, old_max, new_min, new_max: \
            ((tf.stack(t) - old_min) * (new_max - new_min)) / \
            (old_max - old_min) + new_min
        log_fn = lambda s: tf.summary.record_if(
            not tf.math.floormod(s, summ_period))

        if self.r_cfg['train']:
            tsb_writer = tf.summary.create_file_writer(
                str(self.out_folder / str(Folders.TENSORBOARD)))
        else:
            (self.out_folder / str(Folders.RAW_DATA)).mkdir()
            tsb_writer = tf.summary.create_noop_writer()

        db_iter = self.create_db_loder()
        self.curr_img = Path(self.img_paths.pop(0))
        self.patches_ref = self.calc_n_patches(self.curr_img,
                                               self.m_cfg['configs']['patch'][0])
        training = tf.constant(self.r_cfg['train'])
        mse = 0.

        for self.step, data in enumerate(db_iter):
            data_len = len(data)
            batch_diff = self.m_cfg['configs']['batch'] - data_len
            if batch_diff:
                if batch_diff > data_len:
                    num_copies = np.floor(batch_diff / data_len).astype(np.int)
                    data = tf.concat(num_copies * [data], axis=0)
                    data_len = len(data)
                batch_diff = self.m_cfg['configs']['batch'] - data_len
                data = tf.pad(data, [[0, batch_diff]] + 3*[[0, 0]], 'SYMMETRIC')
            t_data = p_transf(data, 0, 1, new_min, new_max)
            with tsb_writer.as_default(), log_fn(self.step):
                with tf.GradientTape(*(2 * [self.r_cfg['train']])) as tape:
                    all_out = self.model(t_data, training=training)
                    inputs, latents, out = all_out
                    outputs = p_transf(out, old_min, old_max, new_min, new_max)
                    if self.m_cfg['configs']['recursive']:
                        inputs = inputs[0]
                    losses = list(map(lambda i, o: self.loss(i, o),
                                      repeat(inputs), outputs))
                    loss_ref = losses
                    if self.m_cfg['configs']['recursive']:
                        loss_ref = [tf.reduce_mean(losses)]

                if self.r_cfg['train']:
                    grads = list(map(lambda l, v: tape.gradient(l, v),
                                     loss_ref, self.vars))
                    list(map(lambda o, g, v: o.apply_gradients(zip(g, v)),
                             self.opt, grads, self.vars))
                    self._log_vars_in_tsb(inputs, outputs, latents, losses,
                                          grads)
                else:
                    outputs = p_transf(outputs, new_min, new_max, 0, 1)
                    self._save_data(outputs, latents)

                self.model.reset_states()

                print('#{:d}: {}'.format(self.step, losses[0].numpy()))
                AutoEnc._clear_last_lines()
                mse += losses[0].numpy()
                if self.step == self.ckpt_iter[0]:
                    tf.keras.experimental.export_saved_model(
                        self.model, self.ckpt_fmt.format(self.ckpt_iter.pop(0)),
                        custom_objects=self.custom_objects)

        print('Average loss: {}'.format(mse/(self.step+1)))
        tsb_writer.close()
