"""Experimental shared-null joint likelihood. No truth or efficacy inputs.

Unlike v0.8, target mixtures and real control blocks share ONE nuisance
parameter vector. Targets may inform that vector through the complete mixture
likelihood; they are never silently treated as known neutral controls.
"""
from itertools import combinations
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp, gammaln


def masks_and_priors(m):
    masks=[np.zeros(m)]; variances=[0.]; classes=[0]; logprior=[0.];moments=[False]
    for j in range(m):
        for variance in (1.,16.):
            masks.append(np.eye(m)[j]);variances.append(variance);classes.append(1);logprior.append(-np.log(2*m));moments.append(variance==1.)
    for k in range(2,m+1):
        subsets=list(combinations(range(m),k))
        for subset in subsets:
            mask=np.zeros(m);mask[list(subset)]=1.
            masks.append(mask);variances.append(1.);classes.append(2);moments.append(True)
            # Maximum-entropy prior over admissible guide subsets, rather than
            # equal cardinalities (which overweights the all-guide subset).
            logprior.append(-np.log(2**m-m-1))
    return np.array(masks),np.array(variances),np.array(classes),np.array(logprior),np.array(moments)


class SharedNullLikelihood:
    """Gaussian rank-one slabs with analytic shared-parameter derivatives.

    Nuisance has exchangeable eigenvariances a (orthogonal to the mean) and b
    (block mean). Measurement matrices remain heteroscedastic and unstructured.
    A continuous nonlocal delta^2*N(0,1) coherent effect is integrated exactly
    per subset. The prior integrates to one and has no point mass at zero.
    """
    def __init__(self,groups,control_values,control_measurement,learn_scale=True,measurement_df=None,measurement_quadrature=0):
        self.learn_scale=learn_scale
        self.measurement_df=measurement_df;self.measurement_quadrature=measurement_quadrature
        self.groups=groups;self.c=np.asarray(control_values);self.cm=np.asarray(control_measurement)
        self.k=self.c.shape[1];self.n=sum(len(x) for x,_ in groups)
        self.spec={x.shape[1]:masks_and_priors(x.shape[1]) for x,_ in groups}

    def density(self,x,measurement,parameters,components=True,derivative=False):
        if not self.measurement_quadrature:
            return self.gaussian_density(x,measurement,parameters,components,derivative)
        from scipy.special import roots_genlaguerre
        df=self.measurement_df[1 if measurement is self.cm else 0]
        nodes,weight=roots_genlaguerre(self.measurement_quadrature,df/2-1)
        weight=weight/weight.sum();scales=(df-2)/(2*nodes)
        n=len(x);q=len(nodes)
        expanded=np.broadcast_to(x[:,None,:],(n,q,x.shape[1])).reshape(n*q,-1)
        matrices=(measurement[:,None,:,:]*scales[None,:,None,None]).reshape(n*q,x.shape[1],x.shape[1])
        result=self.gaussian_density(expanded,matrices,parameters,components,derivative)
        logs,scores=result if derivative else (result,None)
        shape=(n,q,3) if components else (n,q)
        logs=logs.reshape(shape);weighted=logs+np.log(weight).reshape((1,q,1) if components else (1,q))
        integrated=logsumexp(weighted,axis=1)
        if not derivative:return integrated
        dimension=scores.shape[-1];scores=scores.reshape(shape+(dimension,))
        posterior=np.exp(weighted-np.expand_dims(integrated,1))
        return integrated,np.sum(posterior[...,None]*scores,axis=1)

    def gaussian_density(self,x,measurement,parameters,components=True,derivative=False):
        a,b,mu=parameters[:3];n,m=x.shape
        eb=np.ones((m,m))/self.k;ea=np.eye(m)-eb
        cov=measurement+a*ea+b*eb+np.eye(m)*1e-8
        precision=np.linalg.inv(cov);delta=x-mu
        pz=np.einsum('nij,nj->ni',precision,delta)
        clean=-.5*(m*np.log(2*np.pi)+np.linalg.slogdet(cov)[1]+np.sum(delta*pz,axis=1))
        if not components:
            if not derivative:return clean
            scores=np.column_stack([.5*(np.einsum('ni,ij,nj->n',pz,e,pz)-np.einsum('nij,ji->n',precision,e)) for e in (ea,eb)]+[pz.sum(axis=1)])
            return clean,scores
        masks,variances,classes,prior,moments=self.spec[m]
        prior=prior.copy();variances=variances.copy();eta=parameters[5] if len(parameters)>5 else .5
        multi=classes==2;coherent=multi&(masks.sum(axis=1)==m);partial=multi&~coherent
        if len(parameters)>6:variances[multi]=np.exp(parameters[6])
        if partial.any():
            prior[coherent]=np.log(max(eta,1e-300));prior[partial]=np.log(max(1-eta,1e-300))-np.log(partial.sum())
        pm=np.einsum('nij,cj->nci',precision,masks)
        norm=np.einsum('ci,nci->nc',masks,pm)
        proj=pz@masks.T
        factor=variances[None,:]/(1+norm*variances[None,:])
        logs=clean[:,None]+.5*factor*proj**2-.5*np.log1p(norm*variances[None,:])+prior
        effect_mean=factor*proj;effect_second=effect_mean**2+factor
        logs[:,moments]+=np.log(effect_second[:,moments]/variances[None,moments])
        densities=np.column_stack([logsumexp(logs[:,classes==j],axis=1) for j in range(3)])
        if not derivative:return densities
        mean=effect_mean.copy();second=effect_second.copy()
        mean[:,moments]=(effect_mean[:,moments]**3+3*effect_mean[:,moments]*factor[:,moments])/effect_second[:,moments]
        second[:,moments]=(effect_mean[:,moments]**4+6*effect_mean[:,moments]**2*factor[:,moments]+3*factor[:,moments]**2)/effect_second[:,moments]
        scores=[]
        for e in (ea,eb):
            quad=np.einsum('ni,ij,nj->n',pz,e,pz)[:,None]-2*mean*np.einsum('ni,ij,ncj->nc',pz,e,pm)+second*np.einsum('nci,ij,ncj->nc',pm,e,pm)
            trace=np.einsum('nij,ji->n',precision,e)[:,None]
            scores.append(.5*(quad-trace))
        scores.append(pz.sum(axis=1)[:,None]-mean*pm.sum(axis=2))
        scores=np.stack(scores,axis=2)
        class_scores=[]
        for j in range(3):
            use=classes==j;weights=np.exp(logs[:,use]-densities[:,j,None])
            class_scores.append(np.sum(weights[:,:,None]*scores[:,use],axis=1))
        class_scores=np.stack(class_scores,axis=1)
        if partial.any():
            full_log=logs[:,coherent][:,0]-prior[coherent][0]
            partial_log=logsumexp(logs[:,partial]-prior[partial][None,:],axis=1)-np.log(partial.sum())
            eta_score=np.exp(full_log-densities[:,2])-np.exp(partial_log-densities[:,2])
        else:eta_score=np.zeros(n)
        eta_scores=np.zeros((n,3,1));eta_scores[:,2,0]=eta_score
        scale_score=.5*variances[None,:]*proj**2/(1+norm*variances[None,:])**2
        scale_score-=2.5*norm*variances[None,:]/(1+norm*variances[None,:])
        scale_score+=variances[None,:]*(norm+proj**2)/(1+variances[None,:]*(norm+proj**2))
        use=classes==2
        average=np.sum(np.exp(logs[:,use]-densities[:,2,None])*scale_score[:,use],axis=1)
        scale_scores=np.zeros((n,3,1));scale_scores[:,2,0]=average
        return densities,np.concatenate([class_scores,eta_scores,scale_scores],axis=2)

    def evaluate(self,parameters,null=False,gradient=True,local=False,details=False):
        a,b,mu,s=parameters[:4];p=0. if null else parameters[4]
        weights=np.array([(1-p)*(1-s),(1-p)*s,p])
        if gradient:control,cs=self.density(self.c,self.cm,parameters,False,True)
        else:control=self.density(self.c,self.cm,parameters,False,False)
        value=control.sum();grad=np.zeros(len(parameters))
        if gradient:grad[:3]=cs.sum(axis=0)
        locals_=[]
        for x,measurement in self.groups:
            if gradient:logs,scores=self.density(x,measurement,parameters,True,True)
            else:logs=self.density(x,measurement,parameters,True,False)
            logweights=np.full(3,-np.inf)
            np.log(weights,out=logweights,where=weights>0)
            norm=logsumexp(logs+logweights,axis=1)
            post=np.exp(logs+logweights-norm[:,None])
            value+=norm.sum()
            if gradient:
                grad[:3]+=np.einsum('nc,ncj->j',post,scores[:,:,:3])
                if len(parameters)>5:grad[5]+=np.sum(post[:,2]*scores[:,2,3])
                if len(parameters)>6:grad[6]+=np.sum(post[:,2]*scores[:,2,4])
                ratios=np.exp(np.clip(logs-norm[:,None],-700,700))
                grad[3]+=(1-p)*np.sum(ratios[:,1]-ratios[:,0])
                if not null:grad[4]+=np.sum(ratios[:,2]-(1-s)*ratios[:,0]-s*ratios[:,1])
            locals_.append(1-post[:,2])
        # Proper weak regularization in physical variance units; same prior in
        # both models. No truth-dependent variance floor or FDR multiplier.
        value-= (a+b)/16+mu*mu/32
        grad[:3]-=np.array([1/16,1/16,mu/16])
        if len(parameters)>6:
            # Proper truncated N(0,1) prior on log coherent-effect variance.
            from scipy.special import ndtr
            value-=parameters[6]**2/2+.5*np.log(2*np.pi)+np.log(ndtr(4.)-ndtr(-4.))
            grad[6]-=parameters[6]
        if details:return -float(value),np.concatenate(locals_)
        if local:return np.concatenate(locals_)
        return (-float(value),-grad) if gradient else -float(value)

    def fit(self,null=False,start=None):
        observed=np.cov(self.c,rowvar=False);noise=self.cm.mean(axis=0)
        deconvolved=observed-noise
        b=max(1e-5,deconvolved.sum()/self.k)
        a=max(1e-5,(np.trace(deconvolved)-b)/(self.k-1))
        extra=[] if null else [.08,.5]+([0.] if self.learn_scale else [])
        initial=np.array([a,b,0.,.03]+extra) if start is None else np.array(start[:4] if null else start)
        bounds=[(0.,100.),(0.,100.),(-8.,8.),(0.,1.)]+([] if null else [(0.,.5),(0.,1.)]+([(-4.,4.)] if self.learn_scale else []))
        result=minimize(lambda z:self.evaluate(z,null),initial,jac=True,method='L-BFGS-B',bounds=bounds,
                        options=dict(maxiter=150,ftol=1e-10,gtol=1e-5,maxls=30))
        return result

    def integrate_legacy(self,fit,null,draws,seed):
        """Importance-corrected truncated Laplace proposal for joint evidence.

        The fitted likelihood/prior, not the quadratic approximation, weights
        every accepted SHARED parameter draw. Include proposal support mass,
        determinants, normalizing constants, and the full model's p2 prior.
        Finite importance sampling is still approximate; export ESS/MC error.
        """
        theta=fit.x;d=len(theta);hessian=np.zeros((d,d))
        for j in range(d):
            step=1e-4*max(1.,abs(theta[j]));hi=theta.copy();lo=theta.copy();hi[j]+=step;lo[j]-=step
            # One-sided differences at parameter boundaries.
            if j in (0,1,3,4,5) and lo[j]<0:lo[j]=theta[j]
            if j in (3,5) and hi[j]>1:hi[j]=theta[j]
            if j==4 and hi[j]>.5:hi[j]=theta[j]
            hessian[:,j]=(self.evaluate(hi,null)[1]-self.evaluate(lo,null)[1])/(hi[j]-lo[j])
        hessian=(hessian+hessian.T)/2
        # A uniform eta prior has no curvature at p=0. Add curvature to the
        # PROPOSAL ONLY so that an unidentified efficacy mix is not assigned
        # variance1000; importance weights still use the original uniform prior.
        if not null:hessian[5,5]+=12.
        ev,vec=np.linalg.eigh(hessian)
        covariance=(vec/np.maximum(ev,1e-3))@vec.T
        rng=np.random.default_rng(seed);accepted=[];attempts=0
        # Separate, fixed-size support-mass estimate; do not infer proposal
        # normalization from a success-stopped negative-binomial sample.
        proposal=rng.multivariate_normal(theta,covariance,size=32768)
        valid=(proposal[:,:2].min(axis=1)>=0)&(proposal[:,:2].max(axis=1)<=100)&(np.abs(proposal[:,2])<=8)&(proposal[:,3]>=0)&(proposal[:,3]<=1)
        if not null:valid&=(proposal[:,4]>=0)&(proposal[:,4]<=.5)&(proposal[:,5]>=0)&(proposal[:,5]<=1)
        support_probability=float(np.mean(valid))
        if support_probability==0:raise ValueError('Shared-null Laplace draws failed to remain in parameter support')
        # Jointly draw all shared parameters; one draw is shared by EVERY gene.
        while len(accepted)<draws and attempts<draws*1000:
            candidates=rng.multivariate_normal(theta,covariance,size=draws*4)
            for z in candidates:
                attempts+=1
                if min(z[:2])>=0 and max(z[:2])<=100 and abs(z[2])<=8 and 0<=z[3]<=1 and (null or (0<=z[4]<=.5 and 0<=z[5]<=1)):
                    accepted.append(z)
                    if len(accepted)==draws:break
        if len(accepted)!=draws:raise ValueError('Shared-null Laplace draws failed to remain in parameter support')
        objectives=[];locals_=[]
        for z in accepted:
            objective,local=self.evaluate(z,null,gradient=False,details=True)
            objectives.append(objective);locals_.append(local)
        precision=(vec*np.maximum(ev,1e-3))@vec.T
        delta=np.array(accepted)-theta
        logweights=-np.array(objectives)+fit.fun+.5*np.einsum('ni,ij,nj->n',delta,precision,delta)
        weights=np.exp(logweights-logsumexp(logweights))
        local=np.einsum('n,ng->g',weights,np.array(locals_))
        logz=(-fit.fun+d/2*np.log(2*np.pi)+.5*np.linalg.slogdet(covariance)[1]
              +np.log(support_probability)+logsumexp(logweights)-np.log(draws)+(0. if null else np.log(2.)))
        ess=float(1/np.sum(weights**2))
        return local,logz,dict(parameters=theta.tolist(),parameter_covariance=covariance.tolist(),
            hessian_min_eigenvalue=float(ev.min()),draws=draws,draw_attempts=attempts,proposal_support_probability=support_probability,
            proposal_support_probability_mc_se=float(np.sqrt(support_probability*(1-support_probability)/32768)),
            importance_ess=ess,importance_ess_fraction=ess/draws,
            log_evidence_mc_se=float(np.sqrt(max(0.,draws/ess-1)/draws+(1-support_probability)/(32768*support_probability))),log_evidence=float(logz))

    @staticmethod
    def transform(u):
        """Smooth bijection R^d -> physical interior, with exact Jacobian.

        All priors are STILL the physical-coordinate priors of v0.9. The
        Jacobian is not a new regularizer or a variance floor.
        """
        u=np.asarray(u);low=np.array([0.,0.,-8.,0.,0.,0.,-4.])[:len(u)]
        width=np.array([100.,100.,16.,1.,.5,1.,8.])[:len(u)]
        z=expit(u);theta=low+width*z
        logjac=np.sum(np.log(width)-np.logaddexp(0.,-u)-np.logaddexp(0.,u))
        return theta,float(logjac),width*z*(1-z),1-2*z

    def transformed(self,u,null,gradient=True,details=False):
        theta,logjac,jac,dlogjac=self.transform(u)
        if details:
            objective,local=self.evaluate(theta,null,gradient=False,details=True)
            return objective-logjac,local
        objective,g=self.evaluate(theta,null)
        return objective-logjac,g*jac-dlogjac

    def integrate(self,fit,null,draws,seed):
        """Untruncated defensive Student-t importance integration.

        Proposals cover R^d; log-Jacobian and the EXACT joint density correct
        every draw. No rejection/support-mass estimator or fitted p-value.
        A broad defensive component limits missing-tail importance weights.
        Finite ESS remains a diagnostic, not a proof of quadrature accuracy.
        """
        theta=fit.x;d=len(theta)
        low=np.array([0.,0.,-8.,0.,0.,0.,-4.])[:d]
        width=np.array([100.,100.,16.,1.,.5,1.,8.])[:d]
        probability=np.clip((theta-low)/width,1e-4,1-1e-4)
        start=np.log(probability)-np.log1p(-probability)
        transformed_fit=minimize(lambda u:self.transformed(u,null),start,jac=True,
            method='BFGS',options=dict(maxiter=160,gtol=1e-5))
        mode=transformed_fit.x;hessian=np.zeros((d,d))
        for j in range(d):
            step=1e-4*max(1.,abs(mode[j]));hi=mode.copy();lo=mode.copy()
            hi[j]+=step;lo[j]-=step
            hessian[:,j]=(self.transformed(hi,null)[1]-self.transformed(lo,null)[1])/(2*step)
        hessian=(hessian+hessian.T)/2;ev,vec=np.linalg.eigh(hessian)
        covariance=(vec/np.maximum(ev,.05))@vec.T
        rng=np.random.default_rng(seed);df=8.
        component=rng.random(draws)<.1;scales=np.where(component,4.,1.)
        normal=rng.multivariate_normal(np.zeros(d),covariance,size=draws)
        delta=normal*np.sqrt(scales*df/rng.chisquare(df,size=draws))[:,None]
        proposals=mode+delta;precision=(vec*np.maximum(ev,.05))@vec.T
        distance=np.einsum('ni,ij,nj->n',delta,precision,delta)
        logdet=np.linalg.slogdet(covariance)[1]
        constant=gammaln((df+d)/2)-gammaln(df/2)-d/2*np.log(df*np.pi)-.5*logdet
        logq=logsumexp(np.column_stack([
            np.log(.9)+constant-(df+d)/2*np.log1p(distance/df),
            np.log(.1)+constant-d/2*np.log(4.)-(df+d)/2*np.log1p(distance/(4*df))]),axis=1)
        objectives=[];locals_=[]
        for u in proposals:
            objective,local=self.transformed(u,null,details=True)
            objectives.append(objective);locals_.append(local)
        logweights=-np.array(objectives)-logq
        if not np.isfinite(logweights).any():raise ValueError('Stable integration has no finite importance weights')
        weights=np.exp(logweights-logsumexp(logweights));ess=float(1/np.sum(weights**2))
        logz=float(logsumexp(logweights)-np.log(draws)+(0. if null else np.log(2.)))
        local=np.einsum('n,ng->g',weights,np.array(locals_))
        return local,logz,dict(parameters=self.transform(mode)[0].tolist(),
            transformed_mode=mode.tolist(),parameter_covariance=covariance.tolist(),
            hessian_min_eigenvalue=float(ev.min()),proposal_eigen_floor=.05,
            draws=draws,draw_attempts=draws,proposal_support_probability=1.,
            importance_ess=ess,importance_ess_fraction=ess/draws,
            log_evidence_mc_se=float(np.sqrt(max(0.,draws/ess-1)/draws)),
            log_evidence=logz,transformed_optimizer_success=bool(transformed_fit.success),
            transformed_gradient_max=float(np.abs(transformed_fit.jac).max()),
            integration='Exact logit Jacobian; defensive .9*t8(C)+.1*t8(4C) importance sampling; no boundary rejection')

    def decide(self,fdr=.05,draws=128,seed=270901):
        null=self.fit(True);full=self.fit(False,start=np.r_[null.x,.08,.5,([0.] if self.learn_scale else [])])
        _,null_evidence,null_audit=self.integrate(null,True,draws,seed+1)
        local,full_evidence,full_audit=self.integrate(full,False,draws,seed)
        log_bf=float(full_evidence-null_evidence);model_probability=float(expit(log_bf))
        local=1-model_probability*(1-local)
        order=np.argsort(local,kind='stable');q=np.empty_like(local)
        q[order]=np.cumsum(local[order])/np.arange(1,len(local)+1)
        theta=full.x
        return local,q,dict(estimated_alternative_prevalence=float(theta[4]),estimated_single_association_prevalence=float((1-theta[4])*theta[3]),
            learned_effect_variance=bool(self.learn_scale),estimated_effect_variance=float(np.exp(theta[6])) if self.learn_scale else 1.,
            estimated_all_guide_fraction=float(theta[5]),
            shared_parameters=dict(orthogonal_variance=float(theta[0]),mean_eigenvariance=float(theta[1]),reference_shift=float(theta[2])),
            null_parameters=null.x.tolist(),full_objective=float(full.fun),null_objective=float(null.fun),
            full_integration=full_audit,null_integration=null_audit,
            log_model_bayes_factor=log_bf,multi_model_probability=model_probability,
            global_gate_pass=True,em_converged=bool(full.success and null.success),
            optimization_messages=[str(null.message),str(full.message)],null_draws=draws,
            family='Shared-null importance-integrated approximate posterior expected FDP, NOT BH or guaranteed frequentist FDR',
            target_use='Complete clean/single/multi likelihood jointly informs SHARED nuisance, not just prevalence',
            uncertainty='One joint shared parameter draw per gene family; transformed defensive Student-t importance integration with exact Jacobian and explicit no-multi model integration')
