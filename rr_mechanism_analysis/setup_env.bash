

setup()
{
    sudo apt update && sudo apt install \
        ros-jazzy-joint-state-publisher-gui \
        tmux -y

    sudo mkdir -p /etc/apt/keyrings
    curl http://robotpkg.openrobots.org/packages/debian/robotpkg.asc \
        | sudo tee /etc/apt/keyrings/robotpkg.asc
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/robotpkg.asc] http://robotpkg.openrobots.org/packages/debian/pub $(lsb_release -cs) robotpkg" \
        | sudo tee /etc/apt/sources.list.d/robotpkg.list
    sudo apt update
    sudo apt install -qqy robotpkg-py3*-pinocchio -y

    export PATH=/opt/openrobots/bin:$PATH
    export PKG_CONFIG_PATH=/opt/openrobots/lib/pkgconfig:$PKG_CONFIG_PATH
    export LD_LIBRARY_PATH=/opt/openrobots/lib:$LD_LIBRARY_PATH
    export PYTHONPATH=/opt/openrobots/lib/python3.10/site-packages:$PYTHONPATH # Adapt your desired python version here
    export CMAKE_PREFIX_PATH=/opt/openrobots:$CMAKE_PREFIX_PATH
}

launch()
{
    ros2 launch rr_mechanism rr_mechanism.launch.py
}

build()
{
    colcon build --merge-install --symlink-install
}

