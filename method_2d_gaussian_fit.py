import numpy as np
import scipy.optimize as opt
import sys

np.set_printoptions(precision=3)


# define model function and pass independent variables x and y as a list
def two_dim_gaussian(xy_mesh, amplitude, xo, yo, sigma_x, sigma_y, offset, theta):
    xo, yo = float(xo), float(yo)
    x, y = xy_mesh[0], xy_mesh[1]
    a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
    res = offset + amplitude * np.exp(- (a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo) ** 2)))
    return res.ravel()


def two_dim_gaussian_fit(pic_path, pic, verbose=False, bgd_pic=None):
    two_dim_gaussian_prams = two_dim_gaussian.__code__.co_varnames[1:two_dim_gaussian.__code__.co_argcount]

    def find_start_parameter_wo_theta(image):
        # This function should find the start parameter for a 2D_gaussian fit
        # Parameters: amplitude, xo, yo, sigma_x, sigma_y, offset
        # Theta must be rotated manually during the try-fit process
        if image.ndim < 2 or 0 == image.size:
            print("Error: Wrong image dimension: {}; or image size is zero".format(image.ndim))
            print("Abort finding start parameters")
            return None

        try:
            # image size: x = cols, y = rows
            sz_x, sz_y = image.shape[1], image.shape[0]

            # Maximum as amplitude
            amplitude, offset = np.max(img), np.min(img)

            # Location of 2D maximum as xo, yo
            amp_xo, amp_yo = np.array(np.unravel_index(np.argmax(image), image.shape))[::-1]

            # Standard deviation from maximum location
            sigma_x, sigma_y = map(lambda x: np.round(x * (1 - 0.69)), [amp_xo, amp_yo])

            # Gather start fit parameter for returning
            res = [amplitude, amp_xo, amp_yo, sigma_x, sigma_y, offset]

        except:
            print("Warning: Finding suitable start parameters failed!")
            res = None

        return res

    def abort_script():
        print("Leaving script")
        sys.exit()

    def print_gauss_params(param_values, param_errors=None, title="Gaussian Parameter", verbose=verbose):
        if not verbose:
            return
        if param_errors is not None:
            param_zip = zip(two_dim_gaussian_prams, param_values, np.diagonal(param_errors))
        else:
            num = len(param_values)
            param_errors = np.empty(num)
            param_errors.fill(None)
            param_zip = zip(two_dim_gaussian_prams, param_values, param_errors)

        print("===== {} =====".format(title))
        for param, fit_value, fit_error in param_zip:
            if param == "theta":
                unit = "rad"
            elif param == "amplitude":
                unit = "bits"
            else:
                unit = "px"
            if param.startswith("sigma_x"): print("===== Beam size =====")
            rel_error = fit_error / fit_value
            print("{} = ({:.2g} +/- {:.2g}) {} ({:.2f}%)".format(param, fit_value,
                                                                 fit_error, unit,
                                                                 rel_error * 100))
            if param.startswith("sigma_y"): print("====================")

        print("====================")

    # #################################################################################
    # #################################################################################
    # Miscellaneous variables
    f_ending = ".npz"
    if pic.endswith(".npz"):
        pic = pic[:-4]

    # Plot images
    with np.load("{}{}{}".format(pic_path, pic, f_ending)) as data:
        # if bgd_img is not None:
        # img = sum(data["img"][:, :, i] for i in range(3)) - bgd_img
        # else:
        # img = sum(data["img"][:, :, i] for i in range(3))
        if data["img"].ndim == 3:  # Check dimension of loaded picture
            # Image from read data and as sum of all RGB channels
            img = sum(data["img"][:, :, i] for i in range(3))
        else:
            img = data["img"]
        print("Success: Reading image {}{}{} completed".format(pic_path, pic, f_ending))

        if np.max(img) >= 1024:
            print("PiCamera Warning: Image saturated: image maximum = {} >= 1024 pixels".format(np.max(img)))

    # #################################################################################
    # #################################################################################
    # Calculations
    # Size of image
    sx, sy = img.shape[1], img.shape[0]
    print("Image size in pixels: {}".format(img.shape))

    # Start fit  amplitude, xo, yo, sigma_x, sigma_y, offset, theta
    start_param = find_start_parameter_wo_theta(img)
    if start_param is not None:
        print("Start fit parameter found: {}".format(start_param))
        print_gauss_params(start_param, None, "Start parameter set")
    else:
        print("Error: No start fit parameter found, aborting.")
        abort_script()

    # Creating mesh grid based on image size
    x, y = np.linspace(0, sx, sx), np.linspace(0, sy, sy)
    mesh_xy = np.meshgrid(x, y)

    # Try fitting the image with 2D Gaussian model and the found start parameter
    # Theta
    # theta_range = [0, 90]  # Set range to try fitting in degree
    # theta_step = 15  # in degree

    # Start fitting
    # for theta in range(theta_range[1], theta_range[0] - theta_step, -theta_step):
    for theta in [90]:
        # Add latest theta to start fit parameter list
        fit_param = [*start_param, np.radians(theta)]
        # Fit constrains
        boundaries = ([0, 0, 0, -np.inf, -np.inf, 0, 0],
                      [np.inf, sx, sy, np.inf, np.inf, np.max(img), np.radians(180)])
        # boundaries = (-np.inf, np.inf)
        print("Try fitting with theta = {} deg = {:.2f} rad".format(theta, np.radians(theta)))
        try:
            img_popt, img_pcov = opt.curve_fit(two_dim_gaussian, mesh_xy,
                                               img.ravel(), p0=fit_param,
                                               bounds=boundaries)

        except ValueError:
            print("Error while fitting data: ValueError")
            abort_script()

        except RuntimeError:
            print("")

        except opt.OptimizeWarning:
            print("Warning: Covariance of the parameters can not be estimated")
            print("Warning: Maybe the start sigmas need to be changed")
            break

        else:  # If try-statement is successful
            # Print results
            print_gauss_params(img_popt, img_pcov, "Final parameter set")
            # Plot resulting fit
            img_fit = two_dim_gaussian(mesh_xy, *img_popt)
            break  # Break for loop if fit was successful

    print("Done!")

    # Return results
    fit_zip = zip(img_popt, np.diagonal(img_pcov))
    fit_result_parameter_dict = dict(zip(two_dim_gaussian_prams, fit_zip))
    return img, img_fit.reshape(sy, sx), fit_result_parameter_dict, mesh_xy, img_popt, img_pcov

# Not yet included
# Background picture
# if os.path.exists(bgd_pic):
# with np.load("{}".format(bgd_pic)) as data:
# print("bgd picture found")
# bgd_img = sum(data["img"][:, :, i] for i in range(3))
# bgd_fig, bgd_ax = plt.subplots()
# bgd_fig.canvas.set_window_title('Background Picture')
# bgd_im = my_axes(bgd_ax, bgd_img, "")
# bgd_ax.tick_params(labelsize=10)
# bgd_cb = plt.colorbar(bgd_im, ax=bgd_ax)
# bgd_cb.ax.tick_params(labelsize=10)
# else:
# print("No Background picture found")
# bgd_img = None
