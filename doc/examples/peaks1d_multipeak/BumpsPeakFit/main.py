import sys


def setup_bumps():
    """
    Install the refl1d plugin into bumps, but don't run main.
    """
    # vfs_init must be called pretty much first since it replaces filesystem
    # calls with filesystem hooks.  Any module that imports the calls directly
    # e.g., using *from os.path import exists*, needs to have the hook in
    # place before the module is even imported.
    try:
        from bumps.vfs import vfs_init
        vfs_init()
    except ImportError as e:
        # CRUFT: older bumpsdoesn't provide vfs
        print(e)
        pass
    import bumps.cli
    # bumps.cli.set_mplconfig(appdatadir='Refl1D-'+__version__)
    sys.path.append('.')
    import fitplugin
    bumps.cli.install_plugin(fitplugin)


def cli():
    """
    Install the Refl1D plugin into bumps and run the command line interface.
    """
    setup_bumps()

    # # TODO: Add subcommand support to bumps.
    # if len(sys.argv) > 2 and sys.argv[1] == 'align':
    #     from .errors import run_errors
    #     del sys.argv[1]
    #     run_errors()
    # else:
    import bumps.cli
    bumps.cli.main()


def gui():
    """
    Install the Refl1D plugin into bumps and run the graphical user interface.
    """
    setup_bumps()
    import bumps.gui.gui_app
    bumps.gui.gui_app.main()


if __name__ == "__main__":
    cli()
