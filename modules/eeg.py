#!C:\Users\melod\AppData\Local\Programs\Python\Python310\python.exe
# -*- coding: utf-8 -*-

"""
Enhanced EEG Module for Spatial Navigation EEG Experiment
========================================================

This module provides comprehensive EEG integration with detailed event marking
optimized for BrainVision systems and mu rhythm/frontal coherence analysis.

Supports multiple trigger delivery methods:
- Lab Streaming Layer (LSL)
- Parallel port 
- BrainVision Remote Control Server
- BrainVision TriggerBox (USB)
- TCP/UDP markers

Author: Claude
Date: March 2025
"""

import logging
import json
import time
import struct
import os
import numpy as np
import socket
from datetime import datetime
from threading import Lock

# Define ADVANCED_LOGGING_AVAILABLE flag
try:
    from modules.advanced_logging import log_error, log_eeg_event
    ADVANCED_LOGGING_AVAILABLE = True
except ImportError:
    ADVANCED_LOGGING_AVAILABLE = False
    logging.warning("advanced_logging module not found. Using standard logging.")

# Fix the LSL import
try:
    from pylsl import StreamInfo, StreamOutlet, local_clock
    LSL_AVAILABLE = True
except ImportError:
    LSL_AVAILABLE = False
    logging.warning("pylsl module not found. LSL triggers will not be available.")

# Fix the parallel import
try:
    from psychopy import parallel
    PARALLEL_AVAILABLE = True
except ImportError:
    PARALLEL_AVAILABLE = False
    logging.warning("parallel module not found. Parallel port triggers will not be available.")

# Global variables for tracking
trigger_log = []
trigger_lock = Lock()  # Thread safety for trigger logging

# Default BrainVision Remote Control Server settings
DEFAULT_RCS_HOST = "127.0.0.1"
DEFAULT_RCS_PORT = 6700

# Add this helper function near the top of the file
def safe_log_error(loggers, message, exception=None):
    """Safely log an error using advanced logging if available, otherwise use standard logging
    
    Args:
        loggers: Dictionary of logger objects
        message: Error message
        exception: Optional exception object
    """
    logging.error(f"{message}: {str(exception) if exception else ''}")
    
    if loggers and 'error' in loggers and ADVANCED_LOGGING_AVAILABLE:
        try:
            from modules.advanced_logging import log_error
            log_error(loggers, message, exception)
        except Exception as e:
            logging.error(f"Failed to use advanced logging: {e}")


class BrainVisionRCS:
    """BrainVision Remote Control Server interface"""
    
    def __init__(self, host=DEFAULT_RCS_HOST, port=DEFAULT_RCS_PORT):
        """Initialize the RCS connection
        
        Args:
            host: RCS server hostname or IP
            port: RCS server port
        """
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
    
    def connect(self):
        """Connect to BrainVision Remote Control Server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.connected = True
            logging.info(f"Connected to BrainVision RCS at {self.host}:{self.port}")
            return True
        except Exception as e:
            logging.error(f"Failed to connect to BrainVision RCS: {e}")
            self.connected = False
            return False
    
    def send_trigger(self, trigger_code, annotation=None):
        """Send a trigger through the Remote Control Server
        
        Args:
            trigger_code: Numeric trigger code
            annotation: Optional string annotation
            
        Returns:
            bool: Success status
        """
        if not self.connected:
            if not self.connect():
                return False
        
        try:
            # Format: "Trigger S99" or "Annotation text here"
            if annotation:
                command = f"Annotation {annotation}\n"
            else:
                command = f"Trigger S{trigger_code}\n"
            
            self.socket.sendall(command.encode('utf-8'))
            return True
        
        except Exception as e:
            logging.error(f"Failed to send trigger to BrainVision RCS: {e}")
            self.connected = False
            return False
    
    def close(self):
        """Close the connection to the RCS server"""
        if self.socket:
            try:
                self.socket.close()
                self.connected = False
                logging.info("Disconnected from BrainVision RCS")
            except Exception as e:
                logging.error(f"Error closing BrainVision RCS connection: {e}")


class EEGMarkerSystem:
    """Enhanced EEG marker system optimized for BrainVision recording"""
    
    def __init__(self, config, loggers=None):
        """Initialize the EEG marker system
        
        Args:
            config: The experiment configuration object
            loggers: Optional dictionary of logger objects from advanced_logging
        """
        self.config = config
        self.active_systems = []
        self.trigger_log_path = os.path.join(config.data_dir, f'eeg_triggers_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        self.last_trigger_time = 0
        self.trigger_count = 0
        self.loggers = loggers  # Store loggers for use in send_trigger
        
        # Track timing precision
        self.timing_stats = {
            'intervals': [],
            'min_interval': float('inf'),
            'max_interval': 0,
            'mean_interval': 0
        }
        
        # Initialize available systems based on config
        self._init_systems()
        
        # Log initialization
        system_names = [system['name'] for system in self.active_systems]
        if system_names:
            logging.info(f"EEG marker systems initialized: {', '.join(system_names)}")
        else:
            logging.warning("No EEG marker systems initialized")
    
    def _init_systems(self):
        """Initialize available EEG systems based on configuration"""
        # Initialize LSL if available and enabled
        if self.config.use_lsl and LSL_AVAILABLE:
            self._init_lsl()
        
        # Initialize parallel port if available and enabled
        if self.config.use_parallel and PARALLEL_AVAILABLE:
            self._init_parallel()
        
        # Initialize BrainVision Remote Control Server if enabled
        if self.config.use_brainvision_rcs:
            self._init_brainvision_rcs()
        
        # Initialize TCP/UDP markers if enabled
        if self.config.use_tcp_markers:
            self._init_tcp_markers()
        
        # Create local trigger log regardless of other systems
        self._init_local_log()
    
    def _init_lsl(self):
        """Initialize Lab Streaming Layer marker stream"""
        try:
            # Create BrainVision-compatible LSL stream
            info = StreamInfo(
                name="BrainVision RDA Markers" if self.config.brainvision_lsl_name == "" else self.config.brainvision_lsl_name,
                type='Markers',
                channel_count=1,
                nominal_srate=0,
                channel_format='string',  # BrainVision prefers string markers
                source_id='psychopy_spatial_navigation'
            )
            
            # Add experiment metadata to the stream
            desc = info.desc()
            desc.append_child_value("experiment", "Spatial Navigation")
            desc.append_child_value("version", "1.0")
            
            # Create outlet
            outlet = StreamOutlet(info)
            
            # Add to active systems
            self.active_systems.append({
                'name': 'LSL',
                'outlet': outlet,
                'type': 'lsl'
            })
            
            logging.info("LSL marker stream initialized successfully")
        
        except Exception as e:
            logging.error(f"Failed to initialize LSL marker stream: {e}")
    
    def _init_parallel(self):
        """Initialize parallel port for sending triggers"""
        try:
            # Set parallel port address from config
            port_address = self.config.parallel_port_address
            parallel.setPortAddress(port_address)
            
            # Test the port with a simple trigger
            # Send 0 to clear
            parallel.setData(0)
            time.sleep(0.002)  # 2 ms wait
            
            # Send a test value (code 255) to verify connection
            parallel.setData(255)
            time.sleep(0.002)  # 2 ms wait - ensure it's registered
            
            # Clear again
            parallel.setData(0)
            
            # Add to active systems
            self.active_systems.append({
                'name': 'Parallel Port',
                'address': port_address,
                'type': 'parallel',
                'port_object': parallel
            })
            
            logging.info(f"Parallel port initialized successfully at address {hex(port_address)}")
            print(f"TTL trigger system ready: Using parallel port at {hex(port_address)}")
            return True
        
        except Exception as e:
            logging.error(f"Failed to initialize parallel port: {e}")
            print(f"Error initializing TTL trigger system: {e}")
            return False
    
    def _init_brainvision_rcs(self):
        """Initialize BrainVision Remote Control Server connection"""
        try:
            rcs = BrainVisionRCS(
                host=self.config.brainvision_rcs_host,
                port=self.config.brainvision_rcs_port
            )
            
            # Test connection
            if rcs.connect():
                # Add to active systems
                self.active_systems.append({
                    'name': 'BrainVision RCS',
                    'connection': rcs,
                    'type': 'brainvision_rcs'
                })
                
                logging.info(f"BrainVision RCS initialized at {self.config.brainvision_rcs_host}:{self.config.brainvision_rcs_port}")
            else:
                logging.warning("BrainVision RCS connection failed. Will retry when sending triggers.")
        
        except Exception as e:
            logging.error(f"Failed to initialize BrainVision RCS: {e}")
    
    def _init_tcp_markers(self):
        """Initialize custom TCP/UDP marker system"""
        try:
            # Create TCP socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self.config.tcp_marker_host, self.config.tcp_marker_port))
            
            # Add to active systems
            self.active_systems.append({
                'name': 'TCP Markers',
                'socket': sock,
                'type': 'tcp'
            })
            
            logging.info(f"TCP marker system initialized at {self.config.tcp_marker_host}:{self.config.tcp_marker_port}")
        
        except Exception as e:
            logging.error(f"Failed to initialize TCP marker system: {e}")
    
    def _init_local_log(self):
        """Initialize local trigger log file"""
        try:
            # Create header for the log file
            header = {
                'experiment': 'Spatial Navigation EEG',
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'trigger_codes': {k: {'code': v, 'description': self._get_trigger_description(k)} 
                                for k, v in self.config.eeg_trigger_codes.items()},
                'triggers': []
            }
            
            # Write header to file
            with open(self.trigger_log_path, 'w') as f:
                json.dump(header, f, indent=2)
            
            # Add to active systems
            self.active_systems.append({
                'name': 'Local Log',
                'path': self.trigger_log_path,
                'type': 'local_log'
            })
            
            logging.info(f"Local trigger log initialized at {self.trigger_log_path}")
        
        except Exception as e:
            logging.error(f"Failed to initialize local trigger log: {e}")
    
    def _get_trigger_description(self, code_name):
        """Get a human-readable description for a trigger code
        
        Args:
            code_name: Name of the trigger code
            
        Returns:
            str: Description of the trigger code
        """
        # Basic descriptions for standard codes
        descriptions = {
            'experiment_start': 'Experiment begins',
            'experiment_end': 'Experiment ends',
            'block_start': 'Block begins',
            'block_end': 'Block ends',
            'trial_start': 'Trial begins',
            'trial_end': 'Trial ends',
            'fixation_onset': 'Fixation cross appears (baseline)',
            'stimulus_onset': 'Stimulus appears on screen',
            'stimulus_offset': 'Stimulus removed from screen',
            'response': 'Participant makes a response',
            'correct_response': 'Participant response was correct',
            'incorrect_response': 'Participant response was incorrect',
            'no_response': 'Participant did not respond in time',
            'feedback_onset': 'Feedback appears on screen',
            'feedback_offset': 'Feedback disappears from screen',
            'egocentric_condition': 'Egocentric navigation condition',
            'allocentric_condition': 'Allocentric navigation condition',
            'control_condition': 'Control navigation condition',
            'easy_difficulty': 'Easy difficulty level',
            'hard_difficulty': 'Hard difficulty level',
            'control_difficulty': 'Control difficulty level',
            'up_key': 'Up arrow key pressed',
            'down_key': 'Down arrow key pressed',
            'left_key': 'Left arrow key pressed',
            'right_key': 'Right arrow key pressed',
            'coherence_analysis_segment': 'Segment marked for coherence analysis'
        }
        
        # Add descriptions for combined codes
        if code_name.startswith('ego_') or code_name.startswith('allo_') or code_name.startswith('control_'):
            parts = code_name.split('_')
            
            if len(parts) == 3 and parts[2] == 'condition':
                nav_type = 'Egocentric' if parts[0] == 'ego' else 'Allocentric' if parts[0] == 'allo' else 'Control'
                return f"{nav_type} navigation, {parts[1]} difficulty"
            
            elif len(parts) == 3 and parts[2] == 'response':
                nav_type = 'Egocentric' if parts[0] == 'ego' else 'Allocentric' if parts[0] == 'allo' else 'Control'
                return f"{nav_type} {parts[1]} key response"
            
            elif len(parts) == 3 and parts[2] == 'direction':
                nav_type = 'Egocentric' if parts[0] == 'ego' else 'Allocentric' if parts[0] == 'allo' else 'Control'
                return f"{nav_type} {parts[1]} direction"
            
            elif len(parts) == 3 and (parts[2] == 'correct' or parts[2] == 'incorrect'):
                nav_type = 'Egocentric' if parts[0] == 'ego' else 'Allocentric' if parts[0] == 'allo' else 'Control'
                return f"{nav_type} {parts[1]} difficulty, {parts[2]} response"
        
        # Return description if available, otherwise use code name
        return descriptions.get(code_name, code_name.replace('_', ' ').title())
    
    def send_trigger(self, trigger_code, description=None, additional_data=None):
        """Send a trigger code to all active EEG systems
        
        Args:
            trigger_code: The numeric trigger code to send
            description: Optional description for logging
            additional_data: Dictionary of additional data to log
            
        Returns:
            dict: Trigger information including timestamp and success status
        """
        # Initialize trigger info
        trigger_info = {
            'code': trigger_code,
            'description': description or self._get_trigger_description(
                next((k for k, v in self.config.eeg_trigger_codes.items() if v == trigger_code), 'unknown')
            ),
            'timestamp': time.time(),
            'experiment_time': time.time() - self.config.experiment_start_time,
            'lsl_time': local_clock() if LSL_AVAILABLE else None,
            'additional_data': additional_data or {},
            'success': False,
            'systems': []
        }
        
        # Calculate timing interval since last trigger
        if self.last_trigger_time > 0:
            interval = trigger_info['timestamp'] - self.last_trigger_time
            trigger_info['interval'] = interval
            # Update timing statistics
            if len(self.timing_stats['intervals']) == 0 or interval < self.timing_stats['min_interval']:
                self.timing_stats['min_interval'] = interval
            if interval > self.timing_stats['max_interval']:
                self.timing_stats['max_interval'] = interval
            self.timing_stats['intervals'].append(interval)
            if len(self.timing_stats['intervals']) > 0:
                self.timing_stats['mean_interval'] = np.mean(self.timing_stats['intervals'])
        self.last_trigger_time = trigger_info['timestamp']
        self.trigger_count += 1
        trigger_info['sequence'] = self.trigger_count
        
        # No active systems
        if not self.active_systems:
            logging.warning(f"No active EEG systems to send trigger {trigger_code}")
            return trigger_info
        
        success = False
        # Send trigger to each active system
        for system in self.active_systems:
            system_result = self._send_to_system(system, trigger_code, trigger_info)
            trigger_info['systems'].append(system_result)
            if system_result['success']:
                success = True
        
        trigger_info['success'] = success
        
        # Thread-safe logging of trigger
        with trigger_lock:
            global trigger_log
            trigger_log.append(trigger_info)
        
        # Periodically save trigger log to file
        if self.trigger_count % 50 == 0:
            self._save_trigger_log()
        
        # Also log to advanced logging system if available
        if self.loggers and 'eeg' in self.loggers and ADVANCED_LOGGING_AVAILABLE:
            try:
                # Extract block, trial, navigation type and difficulty from additional_data if available
                block_num = additional_data.get('block_num', 0) if additional_data else 0
                trial_num = additional_data.get('trial_num', 0) if additional_data else 0
                nav_type = additional_data.get('navigation_type', 'unknown') if additional_data else 'unknown'
                difficulty = additional_data.get('difficulty', 'unknown') if additional_data else 'unknown'
                
                # Format any remaining additional data as a string
                details = None
                if additional_data:
                    # Filter out the keys we've already used
                    used_keys = {'block_num', 'trial_num', 'navigation_type', 'difficulty'}
                    remaining_data = {k: v for k, v in additional_data.items() if k not in used_keys}
                    if remaining_data:
                        details = str(remaining_data)
                
                # Import here to avoid circular imports
                from modules.advanced_logging import log_eeg_event
                log_eeg_event(
                    self.loggers,
                    trigger_code,
                    description or "Unknown event",
                    block_num,
                    trial_num,
                    nav_type,
                    difficulty,
                    details
                )
            except Exception as e:
                logging.error(f"Failed to log EEG event: {e}")
        
        return trigger_info
    
    def _send_to_system(self, system, trigger_code, trigger_info):
        """Send trigger to a specific EEG system
        
        Args:
            system: System configuration dictionary
            trigger_code: Numeric trigger code to send
            trigger_info: Complete trigger information
            
        Returns:
            dict: Result of the send operation
        """
        system_type = system['type']
        result = {
            'system': system['name'],
            'success': False,
            'error': None
        }
        
        try:
            # Handle LSL
            if system_type == 'lsl':
                # BrainVision prefers marker strings formatted as "S99" or "R99"
                marker_string = f"S{trigger_code}"
                # Add annotation if available
                if trigger_info['description']:
                    marker_string += f",{trigger_info['description']}"
                system['outlet'].push_sample([marker_string])
                result['success'] = True
            
            # Handle parallel port - ENHANCED FOR TTL RELIABILITY
            elif system_type == 'parallel':
                # First send 0 to clear (ensure no lingering signals)
                parallel.setData(0)
                time.sleep(self.config.parallel_reset_delay)  # Configurable delay
                
                # Send trigger code - ensure it's an integer between 0-255
                trigger_value = min(255, max(0, int(trigger_code)))
                parallel.setData(trigger_value)
                
                # Log TTL trigger details
                logging.debug(f"TTL trigger sent: {trigger_value} ({trigger_info['description']})")
                
                # Hold the trigger for configurable delay to ensure it's detected
                time.sleep(self.config.parallel_reset_delay)
                
                # Clear again
                parallel.setData(0)
                result['success'] = True
                result['trigger_value'] = trigger_value
            
            # Handle BrainVision RCS
            elif system_type == 'brainvision_rcs':
                rcs = system['connection']
                # Try to send both a trigger code and an annotation
                success = rcs.send_trigger(trigger_code, trigger_info['description'])
                if not success and not rcs.connected:
                    # Try to reconnect and send again
                    if rcs.connect():
                        success = rcs.send_trigger(trigger_code, trigger_info['description'])
                
                result['success'] = success
            
            # Handle TCP markers
            elif system_type == 'tcp':
                # Format trigger as string (can be customized for specific receiver)
                message = f"TRIGGER:{trigger_code}:{trigger_info['description']}\n"
                system['socket'].sendall(message.encode('utf-8'))
                result['success'] = True
            
            # Handle local log
            elif system_type == 'local_log':
                # Already handled by the main send_trigger method
                result['success'] = True
        
        except Exception as e:
            error_msg = str(e)
            logging.error(f"Failed to send trigger {trigger_code} to {system['name']}: {error_msg}")
            result['error'] = error_msg
        
        return result
    
    def _save_trigger_log(self):
        """Save the current trigger log to file"""
        try:
            # Thread-safe access to trigger log
            with trigger_lock:
                global trigger_log
                
                # Read existing file
                with open(self.trigger_log_path, 'r') as f:
                    log_data = json.load(f)
                
                # Add new triggers
                log_data['triggers'].extend(trigger_log)
                # Add timing statistics
                log_data['timing_stats'] = self.timing_stats
                # Write updated data
                with open(self.trigger_log_path, 'w') as f:
                    json.dump(log_data, f, indent=2)
                
                # Clear the in-memory log
                trigger_log = []
            
            logging.debug(f"Saved {self.trigger_count} triggers to log file")
        except Exception as e:
            logging.error(f"Failed to save trigger log: {e}")
    
    def send_annotation(self, annotation_text):
        """Send a text annotation to the EEG system
        
        This is especially useful for BrainVision recordings to mark
        specific events with detailed text descriptions.
        
        Args:
            annotation_text: The annotation text to send
            
        Returns:
            bool: Success status
        """
        success = False
        for system in self.active_systems:
            system_type = system['type']
            
            try:
                if system_type == 'brainvision_rcs':
                    # Send pure annotation without trigger code
                    rcs = system['connection']
                    if rcs.send_trigger(None, annotation_text):
                        success = True
                
                elif system_type == 'lsl':
                    # Format annotation for LSL
                    system['outlet'].push_sample([f"A:{annotation_text}"])
                    success = True
                
                # Add other system-specific annotation methods as needed
            
            except Exception as e:
                logging.error(f"Failed to send annotation to {system['name']}: {e}")
        
        # Also log the annotation
        annotation_info = {
            'annotation': annotation_text,
            'timestamp': time.time(),
            'experiment_time': time.time() - self.config.experiment_start_time,
        }
        
        with trigger_lock:
            global trigger_log
            trigger_log.append(annotation_info)
        
        return success
    
    def close(self):
        """Close all EEG connections and save final log"""
        # Save final trigger log
        self._save_trigger_log()
        
        # Close each system
        for system in self.active_systems:
            system_type = system['type']
            
            try:
                if system_type == 'brainvision_rcs':
                    system['connection'].close()
                    logging.info("BrainVision RCS connection closed")
                
                elif system_type == 'tcp':
                    system['socket'].close()
                    logging.info("TCP marker connection closed")
                
                # Other system-specific cleanup as needed
            
            except Exception as e:
                logging.error(f"Error closing {system['name']}: {e}")
        
        logging.info("All EEG marker systems closed")
    
    def add_metadata_to_log(self, metadata):
        """Add experiment metadata to the trigger log
        
        Args:
            metadata: Dictionary of metadata to add
            
        Returns:
            bool: Success status
        """
        try:
            # Read existing file
            with open(self.trigger_log_path, 'r') as f:
                log_data = json.load(f)
            
            # Add metadata
            log_data['metadata'] = metadata
            
            # Write updated data
            with open(self.trigger_log_path, 'w') as f:
                json.dump(log_data, f, indent=2)
            
            logging.info(f"Added metadata to trigger log")
            return True
        except Exception as e:
            logging.error(f"Failed to add metadata to trigger log: {e}")
            return False


def setup_eeg(config, loggers=None):
    """Set up the EEG marker system optimized for BrainVision recording
    
    Args:
        config: The experiment configuration object
        loggers: Optional dictionary of logger objects from advanced_logging
        
    Returns:
        EEGMarkerSystem: The initialized EEG marker system
    """
    # Add default configurations if not present
    if not hasattr(config, 'use_lsl'):
        config.use_lsl = True
    
    if not hasattr(config, 'brainvision_lsl_name'):
        config.brainvision_lsl_name = "BrainVision RDA Markers"
    
    if not hasattr(config, 'use_parallel'):
        config.use_parallel = False
    
    if not hasattr(config, 'parallel_port_address'):
        config.parallel_port_address = 0x378  # Default LPT1 address
    
    if not hasattr(config, 'use_brainvision_rcs'):
        config.use_brainvision_rcs = True  # Enable by default for BrainVision
    
    if not hasattr(config, 'brainvision_rcs_host'):
        config.brainvision_rcs_host = DEFAULT_RCS_HOST
    
    if not hasattr(config, 'brainvision_rcs_port'):
        config.brainvision_rcs_port = DEFAULT_RCS_PORT
    
    if not hasattr(config, 'use_tcp_markers'):
        config.use_tcp_markers = False
    
    if not hasattr(config, 'tcp_marker_host'):
        config.tcp_marker_host = "127.0.0.1"
    
    if not hasattr(config, 'tcp_marker_port'):
        config.tcp_marker_port = 5678
    
    # Set experiment start time if not present
    if not hasattr(config, 'experiment_start_time'):
        config.experiment_start_time = time.time()
    
    # Create the marker system with loggers
    try:
        # If BrainVision connection failed in logs, disable it to avoid further errors
        try_brainvision = config.use_brainvision_rcs
        if try_brainvision:
            try:
                test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                test_socket.settimeout(2.0)  # Increased timeout for more reliable check
                try:
                    test_socket.connect((config.brainvision_rcs_host, config.brainvision_rcs_port))
                    test_socket.close()
                except (socket.timeout, ConnectionRefusedError, OSError) as e:
                    logging.warning(f"BrainVision RCS not available ({str(e)}). Continuing without BrainVision markers.")
                    config.use_brainvision_rcs = False
                    
                    # Log this to the advanced logging if available
                    if loggers and 'error' in loggers and ADVANCED_LOGGING_AVAILABLE:
                        from modules.advanced_logging import log_error
                        log_error(loggers, f"BrainVision RCS connection failed: {str(e)}. Continuing without BrainVision markers.")
            except Exception as e:
                logging.warning(f"Failed to check BrainVision RCS: {str(e)}. Continuing without BrainVision triggers.")
                config.use_brainvision_rcs = False
        
        # Create the marker system with the loggers
        marker_system = EEGMarkerSystem(config, loggers)
        
        # Log successful initialization using advanced logging if available
        if loggers and 'system' in loggers:
            active_systems = [system['name'] for system in marker_system.active_systems]
            if active_systems:
                loggers['system'].info(f"EEG marker systems initialized: {', '.join(active_systems)}")
            else:
                loggers['system'].warning("No EEG marker systems initialized")
        
        return marker_system
    except Exception as e:
        logging.error(f"Failed to initialize EEG marker system: {e}")
        
        # Log to advanced logging if available
        if loggers and 'error' in loggers and ADVANCED_LOGGING_AVAILABLE:
            from modules.advanced_logging import log_error
            log_error(loggers, f"Failed to initialize EEG marker system", e)
        
        print("WARNING: Failed to initialize EEG marker system. Continuing without EEG markers.")
        return None


def create_detailed_trigger_codes():
    """Create a comprehensive set of trigger codes for detailed EEG analysis
    
    Returns:
        dict: Dictionary of trigger codes
    """
    # Base trigger codes
    trigger_codes = {
        # Experiment structure
        'experiment_start': 1,
        'experiment_end': 2,
        'block_start': 3,
        'block_end': 4,
        'trial_start': 5,
        'trial_end': 6,
        
        # Stimulus events
        'fixation_onset': 10,
        'stimulus_onset': 11,
        'stimulus_offset': 12,
        
        # Response events
        'response': 20,
        'correct_response': 21,
        'incorrect_response': 22,
        'no_response': 23,
        
        # Feedback events
        'feedback_onset': 30,
        'feedback_offset': 31,
        
        # Navigation conditions (40-49)
        'egocentric_condition': 40,
        'allocentric_condition': 41,
        'control_condition': 42,  # Add missing control condition
        
        # Difficulty levels (50-59)
        'easy_difficulty': 50,
        'hard_difficulty': 51,
        'control_difficulty': 52,
        
        # Response keys (60-69)
        'up_key': 60,
        'down_key': 61,
        'left_key': 62,
        'right_key': 63
    }
    
    # Add combined condition codes (100-199)
    # Format: 1<navigation><difficulty>
    # Where navigation: 0=ego, 1=allo, 2=control
    # Where difficulty: 0=easy, 1=hard, 2=control
    trigger_codes.update({
        'ego_easy_condition': 100,
        'ego_hard_condition': 101,
        'ego_control_condition': 102,
        'allo_easy_condition': 110,
        'allo_hard_condition': 111,
        'allo_control_condition': 112,
        'control_control_condition': 120,  # Add missing control condition
    })
    
    # Add combined response codes (200-299)
    # Format: 2<navigation><response>
    # Where navigation: 0=ego, 1=allo, 2=control
    # Where response: 0=up, 1=down, 2=left, 3=right
    trigger_codes.update({
        'ego_up_response': 200,
        'ego_down_response': 201,
        'ego_left_response': 202,
        'ego_right_response': 203,
        'allo_up_response': 210,
        'allo_down_response': 211,
        'allo_left_response': 212,
        'allo_right_response': 213,
        'control_up_response': 220,
        'control_down_response': 221,
        'control_left_response': 222,
        'control_right_response': 223,
    })
    
    # Add combined navigation direction codes (300-399)
    # These combine the navigation type and direction
    trigger_codes.update({
        'ego_forward_direction': 300,
        'ego_backward_direction': 301,
        'ego_left_direction': 302,
        'ego_right_direction': 303,
        'allo_north_direction': 310,
        'allo_south_direction': 311,
        'allo_west_direction': 312,
        'allo_east_direction': 313,
        'control_forward_direction': 320,
        'control_backward_direction': 321,
        'control_left_direction': 322,
        'control_right_direction': 323,
    })
    
    # Add correct/incorrect outcome combined with condition codes (400-499)
    trigger_codes.update({
        'ego_easy_correct': 400,
        'ego_easy_incorrect': 401,
        'ego_hard_correct': 410,
        'ego_hard_incorrect': 411,
        'ego_control_correct': 420,
        'ego_control_incorrect': 421,
        'allo_easy_correct': 450,
        'allo_easy_incorrect': 451,
        'allo_hard_correct': 460,
        'allo_hard_incorrect': 461,
        'allo_control_correct': 470,
        'allo_control_incorrect': 471,
        'control_control_correct': 480,  # Add missing control outcome codes
        'control_control_incorrect': 481
    })
    
    # Add coherence analysis segment trigger code
    trigger_codes['coherence_analysis_segment'] = 500
    
    # Add Performance + Condition codes (500-599)
    # Format: 5[accuracy][navigation][difficulty]
    # Accuracy: 0=incorrect, 1=correct
    # Navigation: 0=egocentric, 1=allocentric, 2=control
    # Difficulty: 0=easy, 1=hard, 2=control
    accuracies = {'incorrect': 0, 'correct': 1}
    navigations = {'egocentric': 0, 'allocentric': 1, 'control': 2}
    difficulties = {'easy': 0, 'hard': 1, 'control': 2}
    
    for acc_name, acc_code in accuracies.items():
        for nav_name, nav_code in navigations.items():
            for diff_name, diff_code in difficulties.items():
                code = 500 + acc_code * 100 + nav_code * 10 + diff_code
                trigger_codes[f'{acc_name}_{nav_name}_{diff_name}'] = code
    
    return trigger_codes


def test_brainvision_connection(host=DEFAULT_RCS_HOST, port=DEFAULT_RCS_PORT):
    """Test the connection to BrainVision Remote Control Server
    
    Args:
        host: RCS server hostname or IP
        port: RCS server port
        
    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        # Set a reasonable timeout to avoid hanging
        test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_socket.settimeout(2.0)
        test_socket.connect((host, port))
        test_socket.close()
        
        # If we got here, connection was successful, now try the RCS interface
        rcs = BrainVisionRCS(host, port)
        result = rcs.connect()
        if result:
            rcs.send_trigger(99, "Test trigger from PsychoPy")
            rcs.close()
        return result
    except Exception as e:
        logging.error(f"BrainVision connection test failed: {e}")
        return False


def test_ttl_triggers(port_address=0x378, num_triggers=5, interval=0.5):
    """Test TTL trigger functionality by sending test pulses
    
    Args:
        port_address: Parallel port address (default: 0x378)
        num_triggers: Number of test triggers to send
        interval: Interval between triggers in seconds
        
    Returns:
        bool: True if test completed successfully
    """
    try:
        from psychopy import parallel
        
        # Set port address
        parallel.setPortAddress(port_address)
        
        # Clear port initially
        parallel.setData(0)
        time.sleep(0.01)  # 10ms wait
        
        print(f"Sending {num_triggers} test TTL triggers at {interval}s intervals...")
        
        # Send test triggers
        for i in range(num_triggers):
            # Send trigger (value 254 is used for testing - unlikely to conflict with experiment codes)
            trigger_value = 254
            parallel.setData(trigger_value)
            print(f"Test trigger {i+1}/{num_triggers}: Sent value {trigger_value}")
            
            # Delay to ensure it's registered
            time.sleep(0.002)  # 2ms wait
            
            # Clear the trigger
            parallel.setData(0)
            
            # Wait before next trigger
            if i < num_triggers - 1:
                time.sleep(interval)
        
        print("TTL trigger test completed successfully")
        return True
        
    except Exception as e:
        print(f"TTL trigger test failed: {e}")
        logging.error(f"TTL trigger test failed: {e}")
        return False
