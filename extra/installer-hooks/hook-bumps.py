from bumps.gui.resources import resources as gui_resources

# The exported API names are not used within the bumps package, so they
# need to be listed explicitly.
hiddenimports = ["names"]

# Convert [(dir, [f1,f2, ...]), ...] returned by gui_resource.data_files()
# into [(f1, dir), (f2, dir), ...] expected for datas.
datas = [(fname, dirname) for dirname, filelist in gui_resources.data_files() for fname in filelist]
