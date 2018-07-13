import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

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

    @Override
    public INDArray[] predictBatch(String[] images) {
        Model model=this.getModel();
        INDArray input;
        List <INDArray> linputs=new ArrayList<INDArray>();
        Function<String,INDArray> f= model.getPreProcessor();
        for(String image:images)
        {
            input=model.getPreProcessor().apply(image);
            linputs.add(input);
        }
        INDArray []inputs=new INDArray[linputs.size()];
        inputs=linputs.toArray(inputs);
        
        org.deeplearning4j.nn.api.Model m=model.getDeepModel();
        
        if(m instanceof ComputationGraph)
        {
            ComputationGraph graph=(ComputationGraph)m;
            INDArray[] output=graph.output(false,inputs);
            for(int i=0;i<output.length;i++)
            {
                output[i]=Nd4j.sortWithIndices(output[i], 0, false)[0];
            }
            return output;
        }
            
        else
        {
            MultiLayerNetwork graph=(MultiLayerNetwork)m;
            List <INDArray> loutputs=new ArrayList<INDArray>();
            for (INDArray i:inputs)
            {
                INDArray output=graph.output(i);
                output=Nd4j.sortWithIndices(output,0,false)[0];
                loutputs.add(output);
            }
            INDArray []outputs=new INDArray[loutputs.size()];
            outputs=loutputs.toArray(outputs);
            return outputs;
        }
    }
    
}
