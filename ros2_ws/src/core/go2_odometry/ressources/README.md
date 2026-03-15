Motion Capture Instructions
==
> [!NOTE]
> The photos and 3D models are stored using git lfs. If you want to get the files on your PC you will need to install git lfs and run  `git lfs pull` .

## Software setup

#### Additionnal installations
Running the mocap script requires adding two more packages to your local installation:
```bash
# PyKDL for rotation conversions
conda install python-orocos-kdl

# Qualisys Realtime package
python -m pip install pip --upgrade # Upgrade to latest pip
python -m pip install qtm-rt
```

#### Configuration of the 6 DOF body
The parameters for the 6 DOF body to be tracked need to be added to the **Qualisys Track Manager** software (that should normally be present and running in the PC physically connected to the mocap cameras).

The settings for the rigid body can be found in the file **[ressources/Go2-QTM-Body.xml](Go2-QTM-Body.xml)** and can be loaded in the qualisys software by going to *setting panels* -> *6DOF tracking* -> *Load Bodies*.

You should now see an object labeled **Go2**.

---
## Hardware description and building instructions
All parts are 3D printed using PLA and the markers are secured using double sided tape.

> [!IMPORTANT]
> All mocap markers used here are 14mm spherical ones with a ~1m plastic disc base.

### Parts

#### Helmet


| ![](photos/image-1736521381091.png?raw=1) | ![](photos/image-1736521391096.png?raw=1) |
| :------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------- |

##### Files

| File name                 | Internal Object Name | Use |
| :------------------------:| :-----------: | :--------------------------------------: |
| [go2_mocap_attachments.FCStd](CAD/go2_mocap_attachments.FCStd)| MocapHelmet001   | CAD file, modify this one if needed     |
| [go2_mocap_helmet.stl](STL/go2_mocap_helmet.stl)|   /   | STL file                                |


##### Putting on

1. Install 8 motion capture round markers with 15mm diameter plastic base on the slots,  secure with double sided tape between the marker and the part.
2. Attach the helmet to the "head" of the Go2 with 4 M3x20mm screws.

##### Modifying

To modify this part first modify the body called "MocapHelmet001" then convert it to mesh and subtract the mesh called "Head_Scaled1.001". You can then export the mesh as a STL for 3D printing.

#### Rail attachement

| ![](photos/image-1736521515563.png?raw=1) | ![](photos/image-1736521527914.png?raw=1) |
| :------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------- |



##### Files

| File name                 | Internal Object Name | Use |
| :------------------------:| :-----------: | :--------------------------------------: |
| [go2_mocap_attachments.FCStd](CAD/go2_mocap_attachments.FCStd)| rail   | CAD file, modify this one if needed     |
| [go2_mocap_rails.stl](STL/go2_mocap_rails.stl) |  \  | STL file  |


##### Putting on

1. Install 2 M2 nuts in hexagonal slots (see pic) and 2 M2x12mm screws from the other side
   1. If they don't fit try heating them with a heat gun but be careful to not deform the plastic or else the part won't slide in the rails anymore.

<img src="photos/image-1736521593508.png?raw=1" width="610" height="null" />

1. Install 2 motion capture round markers with 15mm diameter plastic base on the slots, secure with double sided tape between the marker and the part.
2. Slide the part in the rail from behind as far as you can and tighten the screws to secure.

##### Modifying

No special precautions

#### Back or "Tail"

##### <img src="photos/image-1736522398812.png?raw=1" width="429" />

##### Files

| File name                 | Internal Object Name | Use |
| :------------------------:| :-----------: | :--------------------------------------: |
| [go2_mocap_attachments.FCStd](CAD/go2_mocap_attachments.FCStd)| back  | CAD file, modify this one if needed     |
| [go2_mocap_back.stl](STL/go2_mocap_back.stl) |   \   | STL file                                |


##### Putting on

1. Install 1 motion capture round markers with 15mm diameter plastic base on the slots,  secure with double sided tape between the marker and the part.
2. Use double sided tape on the hexagonal face and place the part in the slot located in the back of the robot as shown in the image below.

<img src="photos/image-1736522275053.png?raw=1" width="447" height="null" />

##### Modifying

No special precautions
