
/**
 *
 * @author adines
 */
public class PredictorFactory {
    public Predictor getPredictor(String framework, String model)
    {
        ModelFactory modelFactory=new ModelFactory();
        Model m=modelFactory.getModel(framework, model);
        
        return new DL4JPredictor(m);
    }
}
