const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") return new Response("ok", { headers: corsHeaders });

  try {
    const LOVABLE_API_KEY = Deno.env.get("LOVABLE_API_KEY");
    if (!LOVABLE_API_KEY) throw new Error("LOVABLE_API_KEY not configured");

    const body = await req.json();
    const { image, hint, mode } = body ?? {};
    if (!image || typeof image !== "string") {
      return new Response(JSON.stringify({ error: "Missing 'image' (data URL)" }), {
        status: 400,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    // Two modes:
    //  - "scene"  → list ALL animals/objects/people Gemini sees in the full image
    //  - default  → identify the single subject in a cropped image (any animal OR object)
    const isScene = mode === "scene";

    const prompt = isScene
      ? `You are a precise visual recognition expert. Identify EVERY distinct subject visible in this image (animals of any species, people, and notable objects/things — including items COCO-SSD cannot detect like specific animal species, tools, plants, devices, etc.).

Return STRICT JSON with this exact shape and no extra text:
{
  "summary": "one short sentence describing the scene",
  "subjects": [
    {
      "name": "specific common name (e.g. 'Bengal Tiger', 'Red Fox', 'iPhone', 'Acoustic Guitar')",
      "category": "animal" | "person" | "object" | "plant" | "vehicle" | "food" | "other",
      "scientific_name": "Latin binomial if animal/plant, else null",
      "count": integer (how many of this subject are visible),
      "confidence": 0-100 integer,
      "facts": ["short fact 1", "short fact 2"]
    }
  ]
}
Be specific — prefer "Golden Retriever" over "dog", "Honeybee" over "insect".`
      : `You are an expert at identifying animals, objects, and things in images. Identify the MAIN subject in this cropped image — it could be any animal species, a person, or any object/item.${
          hint ? ` Coarse class hint from object detector: "${hint}".` : ""
        }
Return STRICT JSON with this shape and no extra text:
{
  "species": "specific common name (e.g. 'Bengal Tiger', 'Golden Retriever', 'Acoustic Guitar', 'iPhone 15')",
  "scientific_name": "Latin binomial if it's an animal or plant, else null",
  "confidence": 0-100 integer,
  "facts": ["short fact 1", "short fact 2", "short fact 3"]
}
Be as specific as possible (exact breed/species/model). If you genuinely cannot identify it, set species to "Unknown" and confidence 0.`;

    const aiRes = await fetch("https://ai.gateway.lovable.dev/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${LOVABLE_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "google/gemini-2.5-flash",
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: prompt },
              { type: "image_url", image_url: { url: image } },
            ],
          },
        ],
      }),
    });

    if (!aiRes.ok) {
      const txt = await aiRes.text();
      if (aiRes.status === 429) {
        return new Response(JSON.stringify({ error: "Rate limit, try again shortly." }), {
          status: 429, headers: { ...corsHeaders, "Content-Type": "application/json" },
        });
      }
      if (aiRes.status === 402) {
        return new Response(JSON.stringify({ error: "AI credits exhausted." }), {
          status: 402, headers: { ...corsHeaders, "Content-Type": "application/json" },
        });
      }
      throw new Error(`AI gateway ${aiRes.status}: ${txt}`);
    }

    const data = await aiRes.json();
    const raw: string = data?.choices?.[0]?.message?.content ?? "";
    const cleaned = raw.replace(/```json|```/g, "").trim();
    let parsed: unknown = null;
    try { parsed = JSON.parse(cleaned); } catch {
      const m = cleaned.match(/\{[\s\S]*\}/);
      if (m) parsed = JSON.parse(m[0]);
    }

    return new Response(JSON.stringify({ result: parsed, raw }), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  } catch (e) {
    const msg = e instanceof Error ? e.message : "Unknown error";
    console.error("classify-animal error:", msg);
    return new Response(JSON.stringify({ error: msg }), {
      status: 500,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }
});
