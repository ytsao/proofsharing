import sys
import logging


def init_logger(str_filename: str = None):
    global logger
    global stream

    from .utils import initialize_logger, Stream2Logger

    logger = initialize_logger(str_filename=str_filename)
    stream = Stream2Logger(sys.stderr, logger, logging.INFO)


class config:
    # General options
    netname = (None, "the network name, the extension can be only .pyt, .tf and .meta")
    num_tests = (None, "number of images to test")
    relu_transformer = (
        "zonotope",
        "Use Standard zonotope transformer or box transformer",
    )
    dataset = (None, "the dataset, can be either mnist or cifar")

    # L_Infinity attacks
    epsilon = (None, "the epsilon for L_infinity perturbation")
    label = (None, "label of the dataset")

    # Patch attacks
    patch_size = (2, "pxp-size for patch perturbation")

    # Geometric attacks
    data_dir = (None, "data location for geometric analysis")

    # Proof Transfer Online
    template_method = ("l_infinity", "Method for online template creation")
    template_layers = (
        [],
        "Layers the templates are created at. Multiple layers possible.",
    )
    template_domain = ("box", "Domain of the template: box or parallelotope")
    template_with_hyperplanes = (
        False,
        "Allow templates to use hyperplanes, offline only",
    )

    # Proof Transfer Offline
    template_dir = (None, "Path to templates")
    num_templates = (25, "Number of templates should be used at most.")

    # Number of post-conditions should be generated in BaFA framework.
    num_post_cons = (1, "Number of post conditions.")
    vnnlib_file = (None, "vnnlib file location.")

    verifier = ("interval", "choose verification method or abstract domains.")
