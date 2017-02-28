from threading import Thread

import waveform_thread


def on_server_loaded(server_context):
    t = Thread(target=waveform_thread.start_tool, args=())
    t.setDaemon(True)
    t.start()
