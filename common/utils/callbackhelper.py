
def make_reduce_compose(obj, hook_cls, hooks: list):
    """only keeps the overridden methods not the empty ones"""
    def _get_loop_fn(fns):
        def loop(*args, **kwargs):
            for fn in fns:
                fn(*args, **kwargs)
        return loop

    method_list = [func for func in dir(hook_cls)
                   if callable(getattr(hook_cls, func)) and not func.startswith("__")]
    for method in method_list:
        hook_fns = []
        for hook in hooks:
            base_fn = getattr(hook_cls, method)
            hook_fn = getattr(hook, method)
            if hook_fn.__func__ != base_fn:
                hook_fns.append(hook_fn)
        setattr(obj, method, _get_loop_fn(hook_fns))
