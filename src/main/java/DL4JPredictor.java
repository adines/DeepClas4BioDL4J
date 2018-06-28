import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author adines
 */
public class DL4JPredictor extends Predictor{

    public DL4JPredictor(Model model) {
        super(model);
    }

    
    
    @Override
    public String predict(String image) {
        Model model=this.getModel();
        INDArray input=model.getPreProcessor().apply(image);
        
        org.deeplearning4j.nn.api.Model m=model.getDeepModel();
        
        if(m instanceof ComputationGraph)
        {
            ComputationGraph graph=(ComputationGraph)m;
            INDArray[] output=graph.output(false,input);
            return model.getPostProcessor().apply(output[0]);
        }
            
        else
        {
            MultiLayerNetwork graph=(MultiLayerNetwork)m;
            INDArray output=graph.output(input);
            return model.getPostProcessor().apply(output);
        }
    }
    
}
