import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


class EnsembleDisagreement():
    MIXTURE_K_DEFAULT = 20.
    MIXTURE_PRECISION_DEFAULT = 1e-7

    def __init__(self, ensemble_len, img_size, integration_method="simpson", sess=None, dtype=np.float32):
        """ Construct an object for computing ensemble disagreement for probabilistic CARE model.

            Disagreement is computed as H(L) - sum(H(l)), where l is Laplace distribution for an image and L is mixture of ls. Since H(L) can't be computed explicitly it is approximated using Simpson's quadratic formula.

         Args:
            ensemble_len: integer representing number of elements in the ensemble
            img_size: tuple with image size (HxW or ZxHxW). For high depth images (large Z) it is recommended to use EnsembleDisagreement3D
            integration_method: method used for approximating integral in mixture entropy. Possible values are "simpson" and "trapezoidal", default is "simpson"
            sess: optional tf.Session() object, if not provided new session will be created 
            dtype: data type, default is single precision float
        """
        self.__sess = sess if sess is not None else tf.Session()

        with tf.name_scope("ensemble_disagreement"):
            ph_shape = (*img_size, 2)
            self.__img_placeholders = [tf.placeholder(name=f"img_placeholder_{i}", shape=ph_shape, dtype=dtype) for i in range(ensemble_len)]

            self.__mixture_k = tf.placeholder_with_default(self.MIXTURE_K_DEFAULT, shape=(), name="mixture_k")
            self.__mixture_precision = tf.placeholder_with_default(self.MIXTURE_PRECISION_DEFAULT, shape=(), name="mixture_precision")
            self.__integration_method = integration_method
            self.__build_disagreement()

    def eval(self, img_predictions, return_entropies=False, mixture_k=None, mixture_precision=None):
        """ Evaluate ensemble disagreement for provided predictions.

        Args:
            img_predictions: list of Laplace distribution params for each image. Must be of same length as specified when constructing this object (`ensemble_len`). Shape of each item must be (HxWx2) where H and W same as in `img_size` constructor param.
            return_entropies: if True also return values of mixture entropy and Laplace entropies. Default value is False
            mixture_k: number std intervals used when approximating mixture entropy. Default value is MIXTURE_K_DEFAULT
            mixture_precision: precision for numerical integration when approximating mixture entropy. Default value is MIXTURE_PRECISION_DEFAULT
        """
        #mixture_precision: precision for numerical integration when approximating mixture entropy. This influences number of points in the quadratic formula and computation time. Default value is MIXTURE_PRECISION_DEFAULT
        assert len(img_predictions) == len(self.__img_placeholders)
        fd = dict([(self.__img_placeholders[i], img_predictions[i]) for i in range(len(img_predictions))])
        if mixture_k is not None:
            fd[self.__mixture_k] = mixture_k

        if mixture_precision is not None:
            fd[self.__mixture_precision] = mixture_precision

        D, L_e, l_e = self.__sess.run([self.__D, self.__L_entropy, self.__l_entropy], feed_dict=fd)

        if return_entropies:
            return D, L_e, l_e
        else:
            return D

    def __build_disagreement(self):
        """ Build sub graph for computing ensemble disagreement."""
        img_ls = [tfd.Laplace(loc=img_ph[...,0], scale=img_ph[...,1], name=f"laplacian_{i}") for i, img_ph in enumerate(self.__img_placeholders)]
        l_entropies = tf.stack([l.entropy() for l in img_ls], axis=0)
        self.__l_entropy = tf.reduce_mean(l_entropies, axis=0, name="l_entropy")
        self.__L_entropy = self.__build_mixture_entropy(img_ls)
        norm_const = tf.constant(np.log(len(img_ls)) if len(img_ls) > 1 else 1., dtype=self.__l_entropy.dtype)
        self.__D = tf.math.divide(self.__L_entropy - self.__l_entropy, norm_const, name="D")

    def __build_mixture_entropy(self, img_ls):
        """ Build sub graph for entropy for mixture of provided Laplacians (`img_ls`)."""
        with tf.name_scope("mixture_entropy"):
            L_bound_l, L_bound_h = self.__get_integration_bounds(img_ls)
            num_points = self.__get_num_points(L_bound_l, L_bound_h)

            p_logp = lambda ep: self.__get_p_logp(ep, img_ls)

            if self.__integration_method.startswith("simpson"):
                L_entropy = self.__get_simpson_integration(p_logp, L_bound_l, L_bound_h, num_points)
            elif self.__integration_method.startswith("trapez"):
                L_entropy = self.__get_trapezoidal_integration(p_logp, L_bound_l, L_bound_h, num_points)
            else:
                raise ValueError(f"Unsupported integration method {self.__integration_method}.")

            return L_entropy

    def __get_p_logp(self, eval_points, img_ls):
        """ Build sub graph for computing mixture p_logp in 'eval_points', where mixture elements are given as a list 'img_ls'."""
        probs = tf.reduce_mean(tf.stack([l.prob(eval_points) for l in img_ls], axis=0), axis=0)
        log_probs = tf.math.log(probs) / tf.math.log(2.)
        return -probs * log_probs

    def __get_simpson_integration(self, func, bound_l, bound_h, num_points):
        """ Build sub graph for Simpson's integration of function 'func'. """
        eval_points, point_distance = self.__generate_eval_points(bound_l, bound_h, num_points)
        eval_coeffs = self.__compute_simpson_coefficients(eval_points)
        func_vals = func(eval_points)
        return tf.math.multiply((point_distance / 3), tf.reduce_sum(func_vals * eval_coeffs, axis=0), name="L_entropy")

    def __get_trapezoidal_integration(self, func, bound_l, bound_h, num_points):
        """ Build sub graph for trapezoidal integration of function 'func'. """
        eval_points, point_distance = self.__generate_eval_points(bound_l, bound_h, num_points)
        eval_coeffs = self.__compute_trapezoidal_coefficients(eval_points)
        func_vals = func(eval_points)
        return tf.math.multiply((bound_h-bound_l) / num_points, tf.reduce_sum(func_vals * eval_coeffs, axis=0), name="L_entropy")

    def __get_integration_bounds(self, img_ls):
        """ Compute bounds for numerical integration."""
        ls_means = tf.stack([l.mean() for l in img_ls], axis=0)
        ls_scales = tf.stack([l.scale for l in img_ls], axis=0)

        #L_mean = tf.reduce_mean(ls_means, axis=0)
        #L_var = tf.reduce_mean(ls_means ** 2 + ls_scales ** 2 - L_mean ** 2, axis=0)
        #L_std = tf.math.sqrt(L_var)

        #bound_l = L_mean - self.__mixture_k * L_std
        #bound_h = L_mean + self.__mixture_k * L_std

        max_scale = tf.reduce_max(ls_scales)
        bound_l = tf.reduce_min(ls_means, axis=0) - self.__mixture_k * max_scale
        bound_h = tf.reduce_max(ls_means, axis=0) + self.__mixture_k * max_scale
        return bound_l, bound_h

    def __get_num_points(self, start, end):
        # use self.__mixture_precision
        return 200

    def __generate_eval_points(self, start, end, num):
        """ Return equidistant points and distance between them for numeric integration."""
        d = (end - start) / num
        distance = tf.repeat(tf.expand_dims(d, 0), num, axis=0)
        nodes = tf.linspace(0., np.float32(num), num)
        #nodes = tf.reshape(nodes, (nodes.shape[0], 1, 1))
        ones_dim_num = len(distance.shape) - 1
        nodes = tf.reshape(nodes, (nodes.shape[0], *tuple(1 for i in range(ones_dim_num))))
        print(distance, start, nodes)
        return start + nodes * distance, d

    def __compute_simpson_coefficients(self, eval_points):
        """ Return coefficients for Simpson's integration formula."""
        num = eval_points.shape[0]
        coeffs = [2. if i % 2 == 0 else 4. for i in range(num)]
        coeffs[0] = 1.
        coeffs[-1] = 1.
        ones_dim_num = len(eval_points.shape) - 1
        return tf.reshape(coeffs, (int(num), *tuple(1 for i in range(ones_dim_num))))

    def __compute_trapezoidal_coefficients(self, eval_points):
        """ Return coefficients for trapezoidal integration formula."""
        num = eval_points.shape[0]
        coeffs = [1 for i in range(num)]
        coeffs[0] = 0.5
        coeffs[-1] = 0.5
        ones_dim_num = len(eval_points.shape) - 1
        return tf.reshape(coeffs, (int(num), *tuple(1 for i in range(ones_dim_num))))

class EnsembleDisagreement3D(EnsembleDisagreement):

    def __init__(self, ensemble_len, img_size, **kwargs):
        """ Construct an object for computing ensemble disagreement for probabilistic CARE model.
            This class is specialized version of EnsembleDisagreement for volumes. 
            It is more memory efficient as it constructs computation graph for only one 2D plane and uses it for whole z-stack.

         Args:
            ensemble_len: integer representing number of elements in the ensemble
            img_size: tuple with image size (ZxHxW)

            Keyword args are same as in EnsembleDisagreement.
        """
        super(EnsembleDisagreement3D, self).__init__(ensemble_len, (img_size[1], img_size[2]), **kwargs)
        self.__z_dim = img_size[0]

    def eval(self, img_predictions, **kwargs):
        """ Evaluate ensemble disagreement for provided predictions.

        Args:
            img_predictions: list of Laplace distribution params for each image. Must be of same length as specified when constructing this object (`ensemble_len`). Shape of each item must be (ZxHxWx2) where Z, H and W are same as in `img_size` constructor param.

            Keyword args are same as in EnsembleDisagreement.eval().
        """
        super_eval = super(EnsembleDisagreement3D, self).eval
        return_entropies = kwargs["return_entropies"]
        D = []
        L_e = []
        l_e = []

        for i in range(self.__z_dim):
            preds_i = [ip[i,...] for ip in img_predictions]
            res_i = super_eval(preds_i, **kwargs)

            if return_entropies:
                D.append(res_i[0])
                L_e.append(res_i[1])
                l_e.append(res_i[2])
            else:
                D.append(res_i)

        D = np.stack(D, 0)
        if return_entropies:
            return D, np.stack(L_e, 0), np.stack(l_e, 0)
        else:
            return D

def get_ensemble_disagreement(ndim, *args, **kwargs):
    """ Get an object for computing ensemble disagreement.

    Args:
        ndim: number of dimensions of input images (2 or 3)
        args: same as in EnsembleDisagreement and EnsembleDisagreement3D
        kwargs: same as in EnsembleDisagreement and EnsembleDisagreement3D
    """
    if ndim not in [2,3]:
        raise ValueError(f"Images of dimension {ndim} are not suppored (ndim should be 2 or 3).")

    return EnsembleDisagreement3D(*args, **kwargs) if ndim == 3 else EnsembleDisagreement(*args, **kwargs)

def tiled_disagreement(disagreement, tile_size, perc_thr, occ_thr):
    """ For given disagreement map return most unrealiable tiles.
        
        Args:
            tile_size - size of resulting tiles in (z, x, y) (or (x,y)) format
            perc_thr - percentile threshold for disagreement (per z stack)
            occ_thr - occupancy threshold for tiles (float from [0,1], if number of pixels with disagreement higher than 'perc_thr' is above 'occ_thr' tile is marked as unrealiable
    """
    d_size = disagreement.shape
    z = np.arange(0, d_size[0], tile_size[0])
    #x = np.arange(0, d_size[1], int(tile_size[1] / 2))
    #y = np.arange(0, d_size[2], int(tile_size[2] / 2))
    x = np.arange(0, d_size[1], tile_size[1])
    y = np.arange(0, d_size[2], tile_size[2])

    z_stack_percs = np.array([np.percentile(disagreement[i,...], perc_thr) for i in range(d_size[0])])
    z_stack_percs = np.reshape(z_stack_percs, (d_size[0], 1, 1))
    window_thr_disagreement = disagreement > z_stack_percs
    #window_thr_disagreement = disagreement > np.percentile(disagreement, perc_thr)

    for zs in z:
        for xs in x:
            for ys in y:
                window = window_thr_disagreement[zs:zs+tile_size[0], xs:xs+tile_size[1], ys:ys+tile_size[2]]
                window_thr_disagreement[zs:zs+tile_size[0], xs:xs+tile_size[1], ys:ys+tile_size[2]] = np.sum(window) / np.prod(window.shape) > occ_thr
                #window_thr_disagreement[zs:zs+tile_size[0], xs:xs+tile_size[1], ys:ys+tile_size[2]] = np.any(window)


    return window_thr_disagreement

