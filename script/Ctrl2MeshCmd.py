import os
import sys
import numpy as np
import maya.cmds as cmds
import maya.api.OpenMaya as OpenMaya

controllerShortFlagName = '-cff'
controllerLongFlagName = '-controllerFileFlag'

controllerpathShortFlagName = '-cpf'
controllerpathLongFlagName = '-controllerpathFlag'

meshShortFlagName = '-mf'
meshLongFlagName = '-meshFlag'

meshpathShortFlagName = '-mpf'
meshpathLongFlagName = '-meshpathFlag'

neutralShortFlagName = "-npf"
neutralpathLongFlagName = "-neutralpathFlag"

startShortFlagName = "-sf"
startLongFlagName = "-startFlag"

endShortFlagName = "-ef"
endLongFlagName = "-endFlag"


def syntaxCreator():
    syntax = OpenMaya.MSyntax()

    syntax.addFlag(controllerShortFlagName, controllerLongFlagName, OpenMaya.MSyntax.kString)
    syntax.addFlag(controllerpathShortFlagName, controllerpathLongFlagName, OpenMaya.MSyntax.kString)
    syntax.addFlag(meshShortFlagName, meshLongFlagName, OpenMaya.MSyntax.kString)
    syntax.addFlag(meshpathShortFlagName, meshpathLongFlagName, OpenMaya.MSyntax.kString)

    syntax.addFlag(startShortFlagName, startLongFlagName, OpenMaya.MSyntax.kUnsigned)
    syntax.addFlag(endShortFlagName, endLongFlagName, OpenMaya.MSyntax.kUnsigned)

    syntax.addFlag(neutralShortFlagName, neutralpathLongFlagName, OpenMaya.MSyntax.kString)

    return syntax


##########################################################
# Plug-in
##########################################################
class Ctrl2MeshCmd(OpenMaya.MPxCommand):
    kPluginCmdName = 'Ctrl2MeshCmd'

    def __init__(self):
        OpenMaya.MPxCommand.__init__(self)
        self.controllerFileName = ""
        self.controllerfilePath = ""
        self.saveController = False

        self.meshnodeName = ""
        self.meshfilePath = ""
        self.saveMesh = False

        self.start = 0
        self.end = int(cmds.findKeyframe(which="end"))

        self.neutralMode = True

        self.neutralfilePath = ""

    @staticmethod
    def cmdCreator():
        return Ctrl2MeshCmd()

    def parseArguments(self, args):
        argData = OpenMaya.MArgParser(self.syntax(), args)

        if argData.isFlagSet(controllerShortFlagName):
            self.controllerFileName = argData.flagArgumentString(controllerShortFlagName, 0)
            self.saveController = True

        if argData.isFlagSet(controllerpathShortFlagName):
            self.controllerfilePath = argData.flagArgumentString(controllerpathShortFlagName, 0)
            self.saveController = True

        if argData.isFlagSet(meshShortFlagName):
            self.meshnodeName = argData.flagArgumentString(meshShortFlagName, 0)
            self.saveMesh = True

        if argData.isFlagSet(meshpathShortFlagName):
            self.meshfilePath = argData.flagArgumentString(meshpathShortFlagName, 0)
            self.saveMesh = True

        if argData.isFlagSet(startShortFlagName):
            self.start = argData.flagArgumentInt(startShortFlagName, 0)

        if argData.isFlagSet(endShortFlagName):
            self.end = argData.flagArgumentInt(endShortFlagName, 0)

        if argData.isFlagSet(neutralShortFlagName):
            self.neutralfilePath = argData.flagArgumentString(neutralShortFlagName, 0)
            self.neutralMode = False

    def getObj(self, string):
        SelectionList = OpenMaya.MSelectionList()
        SelectionList.add(string)
        obj = SelectionList.getDagPath(0)
        return obj

    def getCtrlName(self, controllerFileName):
        plugNameList = []
        with open(controllerFileName) as f:
            for line in f:
                line = line.strip('\n')
                if line == '':
                    continue
                names = line.split(' ')
                coords = ['x', 'y', 'z']
                if names[2] in coords:
                    name = names[0] + '.' + names[1] + names[2]
                else:
                    name = names[0] + '.' + names[2]
                plugNameList.append(name)
        return plugNameList

    def getCtrlValue(self, plugList, length):
        ctrlValue = []
        for i in range(length):
            value = cmds.getAttr(plugList[i])
            ctrlValue.append(round(value, 3))
        return ctrlValue

    def getMeshValue(self, obj):
        print(obj)
        MFnMesh = OpenMaya.MFnMesh(obj)
        points = MFnMesh.getPoints(OpenMaya.MSpace.kObject)
        pointList = []
        for i in points:
            pointList.append(round(i.x, 6))
            pointList.append(round(i.y, 6))
            pointList.append(round(i.z, 6))
        return pointList

    def saveArray(self, array, savepath):
        array = np.array(array).astype(np.float32)
        return np.save(savepath, array)

    def getValuesByFrames(self):

        if self.saveMesh:
            meshobj = self.getObj(self.meshnodeName)

        if not self.neutralMode:
            neutral_head_mesh = np.load(self.neutralfilePath)

        for i in range(self.start, self.end + 1):
            cmds.currentTime(i)
            if self.saveController:
                plugNameList = self.getCtrlName(self.controllerFileName)
                controllerValue = self.getCtrlValue(plugNameList, len(plugNameList))
                self.saveArray(controllerValue, os.path.join(self.controllerfilePath, str(i) + '.npy'))

            if self.saveMesh:
                pointValue = self.getMeshValue(meshobj)
                if not self.neutralMode:
                    self.saveArray((pointValue - neutral_head_mesh), os.path.join(self.meshfilePath, str(i) + '.npy'))
                else:
                    self.saveArray(pointValue, os.path.join(self.meshfilePath, 'neutralHead.npy'))


    def doIt(self, args):
        self.parseArguments(args)
        self.getValuesByFrames()


##########################################################
# Plug-in initialization.
##########################################################

def maya_useNewAPI():
    """
    The presence of this function tells Maya that the plugin produces, and
    expects to be passed, objects created using the Maya Python API 2.0.
    """
    pass


def initializePlugin(mobject):
    mplugin = OpenMaya.MFnPlugin(mobject)
    try:
        mplugin.registerCommand(Ctrl2MeshCmd.kPluginCmdName, Ctrl2MeshCmd.cmdCreator, syntaxCreator)
    except:
        sys.stderr.write('Failed to register command: ' + Ctrl2MeshCmd.kPluginCmdName)


def uninitializePlugin(mobject):
    mplugin = OpenMaya.MFnPlugin(mobject)
    try:
        mplugin.deregisterCommand(Ctrl2MeshCmd.kPluginCmdName)
    except:
        sys.stderr.write('Failed to unregister command: ' + Ctrl2MeshCmd.kPluginCmdName)
