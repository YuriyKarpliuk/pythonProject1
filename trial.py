from dllInteraction import ControllerInterface

def main():
    dll_path = 'D:/4course/2sem/диплом/BSc_APP_1_Output/BSc_APP_1_Libd.dll'
    controller_interface = ControllerInterface(dll_path)
    controller_instance_variable = controller_interface.initialize_controller()
    print("controller_instance_variable: {}".format(controller_instance_variable))
    connected_instance_variable = controller_interface.connect_controller(controller_instance_variable)
    print("connected_instance_variable: {}".format(connected_instance_variable))
    process_data_variable = controller_interface.process_data(controller_instance_variable)
    print("process_data_variable: {}".format(process_data_variable))


if __name__ == "__main__":
    main()
