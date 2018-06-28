/**
 *
 * @author adines
 */
public abstract class Predictor implements IPredictor{
    
    private final Model model;

    public Predictor(Model model) {
        this.model = model;
    }

    public Model getModel() {
        return model;
    }
}
