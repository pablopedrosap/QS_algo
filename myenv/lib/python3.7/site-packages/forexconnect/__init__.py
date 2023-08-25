import os

origin_work_dir = os.getcwd()
here = os.path.abspath(os.path.dirname(__file__))
lib_path = os.path.join(here, "lib")
os.chdir(lib_path)

from forexconnect.lib import fxcorepy
from forexconnect.ForexConnect import ForexConnect
from forexconnect.TableManagerListener import TableManagerListener
from forexconnect.SessionStatusListener import SessionStatusListener
from forexconnect.LiveHistory import LiveHistoryCreator
from forexconnect.EachRowListener import EachRowListener
from forexconnect.ResponseListener import ResponseListener, ResponseListenerAsync
from forexconnect.TableListener import TableListener
from forexconnect.common import Common

fxcorepy.O2GTransport.set_transport_modules_path(lib_path)

os.chdir(origin_work_dir)   