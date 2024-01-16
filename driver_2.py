import sys
from ctypes import *
import time
import gi
import cv2
from multiprocessing.managers import SharedMemoryManager
import multiprocessing as mp
gi.require_version("Gst", "1.0")
gi.require_version("GstRtspServer", "1.0")
from gi.repository import Gst, GstRtspServer, GLib, GObject
import numpy as np 
import threading
import multiprocessing
import logging
Gst.init(None)
from gst_meta import GstOverlayOpenCv
from gst_overlay_opencv import GstOverlayOpenCv
# from gst_ccl_opencv import GstOverlayOpenCv
from gst_buffer_info_meta import get_meta
from check_meta import GstOverlayOpenCv
import traceback
import ghetto_nvds
sys.path.append('/workspace/pyCCL')
import cudaCCLWrapper
import cupy as cp



def on_debug(category, level, dfile, dfctn, dline, source, message, user_data):
    if source:
        logging.info('Debug {} {}: {}'.format(
            Gst.DebugLevel.get_name(level), source.name, message.get()))
    else:
        logging.info('Debug {}: {}'.format(
            Gst.DebugLevel.get_name(level), message.get()))


def set_gst_logger(enable=False , verbosity=None):
    result = True
    try:
        if enable:
            Gst.debug_set_active(True)
            if verbosity and isinstance(verbosity, int) and (verbosity in range(0,10)):
                Gst.debug_set_default_threshold(verbosity)
            else:
                result = False
                logging.error("GST verbosity allowed to be in 0-6")

            Gst.debug_add_log_function(on_debug, None)
            Gst.debug_remove_log_function(Gst.debug_log_default)

        else:
            Gst.debug_set_active(False)
            
    
    except Exception as e:
        result = False

    return result




class Decoder(multiprocessing.Process):

    def __init__(self, gie='nvinfer', codec='H264', bitrate=4000000,udp_port=5400):
        super().__init__()
        self.gie = gie
        self.codec = codec
        self.bitrate = bitrate
        self.udp_port = udp_port

        self.stop_event = multiprocessing.Event()
        self.is_dead_event = multiprocessing.Event()
        self.new_frame_ready = multiprocessing.Event()
        self.overlay_queue = multiprocessing.Queue()
        self.overlay_queue_kill_event = multiprocessing.Event()
        self.gstoverlay = None
        # Glib loop for encoder pipeline to keep alive
        self.gloop = GLib.MainLoop()
        self.gloop_thread = threading.Thread(target=self.gloop.run)
        self.gloop_thread.daemon = True


    def get_new_frame_ready(self):
        return self.new_frame_ready
    
    def get_stop_event(self):
        return self.stop_event
    
    def get_overlay_queue(self):
        return self.overlay_queue

    def get_overlay_queue_kill_event(self):
        return self.overlay_queue_kill_event
    
    def run(self):
        self.pipeline = None
        try:
            self.gloop_thread.start()
        except BaseException as e:
            traceback.print_exc()

        # Create and launch the pipeline and main loop
        result = self.launch_pipeline()

        # Keep live main loop
        time.sleep(500)


        # Cleanup the resources
        self.pipeline.set_state(Gst.State.NULL)
        self.gloop.quit()
        del self.pipeline
        del self.bus
        del self.gloop

        # Wait to finish the main loop
        self.gloop_thread.join()
        
        self.is_dead_event.set()    
        print("[D-{}] FREED ALL RESOURCES !\n".format(self.cam_idx))


    def callback(self,pad, info):
        buf = info.get_buffer()
    
        if buf is None:
            return Gst.PadProbeReturn.OK


        caps = pad.get_current_caps()
        if caps is None:
            return Gst.PadProbeReturn.OK

        # Extract width, height and format from the caps
        width = caps.get_structure(0).get_value('width')
        height = caps.get_structure(0).get_value('height')
        format = caps.get_structure(0).get_value('format')
        # print(width,height,format)
        is_mapped, map_info = buf.map(Gst.MapFlags.READ)
        if is_mapped:
            try:
                source_surface = ghetto_nvds.NvBufSurface(map_info)

                dest_array = cp.zeros(
                    (source_surface.surfaceList[0].height, source_surface.surfaceList[0].width, 4),
                    dtype=cp.uint8
                )

                dest_surface = ghetto_nvds.NvBufSurface(map_info)
                dest_surface.struct_copy_from(source_surface)
                assert(source_surface.numFilled == 1)
                assert(source_surface.surfaceList[0].colorFormat == 19) # RGBA

                dest_surface.surfaceList[0].dataPtr = dest_array.data.ptr

                dest_surface.mem_copy_from(source_surface)
                print(type(dest_array[:,:,0]))
                # print(type(dest_surface.surfaceList[0].dataPtr ))

                boxes = cp.zeros(
                    (source_surface.surfaceList[0].height* source_surface.surfaceList[0].width , 4),
                    dtype=cp.uint16
                )
                cudaCCLWrapper.PyprocessCCL(dest_array[:,:,0],width,height,boxes)
                print(cp.asnumpy(boxes))

                # print(boxes)
            finally:
                buf.unmap(map_info)
        # Extract the data from the buffer and convert to a numpy array
        # data = buf.extract_dup(0, buf.get_size())
        # arr = np.ndarray(shape=(height, width, 1), dtype=np.uint8, buffer=data)
        return Gst.PadProbeReturn.OK


    def launch_pipeline(self,):
        '''
            Configuring the all necessary plugins and setting up and launching the pipeline.
            Run the glib main loop thread to make the process alive

        '''

        Gst.init(None)
        if self.pipeline:
            self.pipeline.set_state( Gst.State.PAUSED)
            time.sleep(.1)
            self.pipeline.set_state( Gst.State.NULL)
            time.sleep(.1)
        # self.pipeline = Gst.parse_launch('videotestsrc pattern=18 ! videoconvert ! xvimagesink')
        # self.pipeline = Gst.parse_launch('videotestsrc pattern=18  \
        #                     ! videoscale ! video/x-raw,width=1920,height=1080 !  videoconvert ! capsfilter ! video/x-raw,format=RGB !  videoconvert name=convert0 \
        #                     ! gstoverlayopencv name=gstoverlay ! checkmeta ! videoconvert ! fpsdisplaysink')
        
        
        self.pipeline = Gst.parse_launch('filesrc location=dance.mp4 ! qtdemux ! h264parse ! avdec_h264  ! nvvideoconvert \
                            ! capsfilter ! video/x-raw(memory:NVMM),format=RGBA ! identity name=convert0 ! nvvideoconvert ! videoconvert ! fpsdisplaysink')

        # Plugin to add inferance
        convert = self.pipeline.get_by_name('convert0')
        convert_sink = convert.get_static_pad('sink')
        convert_sink.add_probe(Gst.PadProbeType.BUFFER, self.callback)

        self.pipeline.set_state(Gst.State.PLAYING)

        # Let the process take some time establish the connection before giving the feedback to end user.
        time.sleep(3)
    
        return True

    

    def stop_stream(self):
         self.stop_event.set()
         return True
    

    def is_dead(self):
        return self.is_dead_event.is_set()
    
if __name__ == "__main__":
    set_gst_logger(True,3)
    decoder = Decoder()
    decoder.daemon = True
    decoder.start()
    decoder.join()

