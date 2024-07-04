from logger.Logger import Logger

logger = Logger().get_logger(__name__)


class OffloadingAlgo:
    def __init__(self,
                 avg_speed: float,
                 num_layers: int,
                 layers_sizes: list,
                 inference_time_device: list,
                 inference_time_edge: list
                 ) -> None:
        self.avg_speed = avg_speed
        self.num_layers = num_layers
        self.layers_sizes = layers_sizes
        self.inference_time_device = inference_time_device
        self.inference_time_edge = inference_time_edge
        self.best_offloading_layer = 0
        self.lowest_evaluation = float('inf')

    @staticmethod
    def evaluation(initial_cost: float, layer_data_size: float, edge_computation_cost: list, avg_speed: float):
        logger.info(f"Performing Evaluation:")
        logger.info(f"DS:           {layer_data_size}")
        logger.info(f"IC:           {initial_cost}")
        logger.info(f"CE:           {edge_computation_cost}")
        logger.info(f"AvgSpeed:     {avg_speed}")
        evaluation = round(initial_cost + (layer_data_size / avg_speed) + edge_computation_cost, 3)
        logger.info(f"Evaluation =  {evaluation}")
        return evaluation

    def edge_only_computation_evaluation(self):
        # Fast offloading: Edge Only Computation
        logger.info(f"Performing Edge Only Offloading:")

        initial_cost = 0
        first_layer_size = self.layers_sizes[0]
        edge_computation_cost = sum(self.inference_time_edge[:self.num_layers + 1])

        self.lowest_evaluation = self.evaluation(
            initial_cost=initial_cost,
            layer_data_size=first_layer_size,
            avg_speed=self.avg_speed,
            edge_computation_cost=edge_computation_cost,
        )

    def mixed_computation_evaluation(self):
        # Partial Offloading: Edge and Device Computation
        logger.info(f"Performing Partial Offloading:")
        for layer in range(0, self.num_layers - 1):
            initial_cost = (0 if layer == 0 else sum(self.inference_time_device[:layer]))
            edge_computation_cost = sum(self.inference_time_edge[layer:self.num_layers])
            layer_data_size = self.layers_sizes[layer + 1]

            evaluation = self.evaluation(
                initial_cost=initial_cost,
                layer_data_size=layer_data_size,
                avg_speed=self.avg_speed,
                edge_computation_cost=edge_computation_cost,
            )

            if evaluation < self.lowest_evaluation:
                self.lowest_evaluation = evaluation
                self.best_offloading_layer = layer

    def device_only_evaluation(self):
        logger.info(f"Performing Device Only Offloading:")
        initial_cost = sum(self.inference_time_device[:self.num_layers + 1])
        layer_data_size = 0
        edge_computation_cost = 0
        # No Offloading: Device Only Computation
        last_evaluation = self.evaluation(
            initial_cost=initial_cost,
            layer_data_size=layer_data_size,
            avg_speed=self.avg_speed,
            edge_computation_cost=edge_computation_cost,
        )
        if last_evaluation < self.lowest_evaluation:
            self.best_offloading_layer = self.num_layers
            self.lowest_evaluation = last_evaluation

    def static_offloading(self):
        logger.info(f"Performing Static Offloading:")
        logger.info(f"Total Neural Network Layers: {self.num_layers}")

        self.edge_only_computation_evaluation()
        self.mixed_computation_evaluation()
        self.device_only_evaluation()

        logger.info("Computation Completed - Offloading Summary:")
        logger.info(f"Lowest Evaluation: {self.lowest_evaluation}")
        logger.info(f"Best Offloading Layer: {self.best_offloading_layer}")

        return self.best_offloading_layer

    def get_info(self):
        return self.__dict__
