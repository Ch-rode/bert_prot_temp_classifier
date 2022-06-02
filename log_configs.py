import logging 
import logging.handlers
import sys
import datetime
import inspect
import time



logging.basicConfig(level=logging.INFO,
                   filename='pipeline_log.log',
                   format='%(asctime)s %(name)s %(levelname)s:%(message)s',
                   datefmt='%H:%M:%S')

logger = logging.getLogger(__name__)

formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', 
                              '%m-%d-%Y %H:%M:%S')


stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)
stderr_handler = logging.StreamHandler(sys.stderr)
stderr_handler.setLevel(logging.DEBUG)
stderr_handler.setFormatter(formatter)
logger.addHandler(stderr_handler)


def fstr(fstring_text, locals, globals=None):
    """
    Dynamically evaluate the provided fstring_text
    """
    locals = locals or {}
    globals = globals or {}
    ret_val = eval(f'f"{fstring_text}"', locals, globals)
    return ret_val
    

class logged(object):
    """
    Decorator class for logging function start, completion, and elapsed time.
    Example of usage: @logged(log_fn=logging.debug)
    @logged("doing a thing with {foo_obj.name}")
    """

    def __init__(
        self,
        desc_text="'{desc_detail}' call to {fn.__name__}()",
        desc_detail="",
        start_msg="Beginning {desc_text}...",
        success_msg="Completed {desc_text}  {elapsed}",
        log_fn=logging.info,
        **addl_kwargs,
    ):
        """ All arguments optional """
        self.context = addl_kwargs.copy()  # start with addl. args
        self.context.update(locals())  # merge all constructor args
        self.context["elapsed"] = None
        self.context["start"] = time.time()

    def re_eval(self, context_key: str):
        """ Evaluate the f-string in self.context[context_key], store back the result """
        self.context[context_key] = fstr(self.context[context_key], locals=self.context)

    def elapsed_str(self):
        """ Return a formatted string, e.g. '(HH:MM:SS elapsed)' """
        seconds = time.time() - self.context["start"]
        return "({} elapsed)".format(str(datetime.timedelta(seconds=int(seconds))))

    def __call__(self, fn):
        """ Call the decorated function """

        def wrapped_fn(*args, **kwargs):
            """
            The decorated function definition. Note that the log needs access to 
            all passed arguments to the decorator, as well as all of the function's
            native args in a dictionary, even if args are not provided by keyword.
            If start_msg is None or success_msg is None, those log entries are skipped.
            """
            self.context["fn"] = fn
            fn_arg_names = inspect.getfullargspec(fn).args
            for x, arg_value in enumerate(args, 0):
                self.context[fn_arg_names[x]] = arg_value
            self.context.update(kwargs)
            desc_detail_fn = None
            log_fn = self.context["log_fn"]
            # If desc_detail is callable, evaluate dynamically (both before and after)
            if callable(self.context["desc_detail"]):
                desc_detail_fn = self.context["desc_detail"]
                self.context["desc_detail"] = desc_detail_fn()

            # Re-evaluate any decorator args which are fstrings
            self.re_eval("desc_detail")
            self.re_eval("desc_text")
            # Remove 'desc_detail' if blank or unused
            self.context["desc_text"] = self.context["desc_text"].replace("'' ", "")
            self.re_eval("start_msg")
            if self.context["start_msg"]:
                # log the start of execution
                log_fn(self.context["start_msg"])
            ret_val = fn(*args, **kwargs)
            if desc_detail_fn:
                # If desc_detail callable, then reevaluate
                self.context["desc_detail"] = desc_detail_fn()
            self.context["elapsed"] = self.elapsed_str()
            # log the end of execution
            log_fn(fstr(self.context["success_msg"], locals=self.context))
            return ret_val

        return wrapped_fn