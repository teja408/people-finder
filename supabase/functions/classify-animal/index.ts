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
      ? `You are a precise visual recognition expert. Identify ONLY the meaningful FOREGROUND subjects in this image. You are restricted to THREE categories:

1. "person"  — humans
2. "animal"  — any animal (dogs, cats, birds, insects, fish, wildlife, etc.)
3. "object"  — discrete, useful man-made things people interact with (e.g. car, bicycle, book, pen, phone, laptop, cup, bottle, chair, bag, watch, guitar, ball, tool, appliance)

STRICTLY EXCLUDE (do NOT label these — never return them):
- Background / scenery: grass, sky, clouds, trees (as scenery), bushes, mountains, water, sand, road, floor, wall, ceiling, ground, dirt
- Decorations / accessories that are not standalone usable items: rings, necklaces, earrings, bracelets, jewelry, hats, clothing, shoes, belts, glasses
- Body parts: hands, faces, hair, eyes
- Generic textures or surfaces: shadows, reflections, patterns
- Food unless it's clearly a discrete prepared item (skip raw ingredients/garnishes)

For EACH individual instance, return a tight bounding box in NORMALIZED coordinates (0.0 to 1.0) where:
- x_min, y_min = top-left corner (0,0 is top-left of image)
- x_max, y_max = bottom-right corner (1,1 is bottom-right)

Return STRICT JSON with this exact shape and no extra text:
{
  "summary": "one short sentence describing the scene",
  "subjects": [
    {
      "name": "specific common name (e.g. 'Bengal Tiger', 'Golden Retriever', 'iPhone', 'Book', 'Pen', 'Car')",
      "category": "person" | "animal" | "object",
      "scientific_name": "Latin binomial if animal, else null",
      "count": integer (how many of this subject are visible),
      "confidence": 0-100 integer,
      "facts": ["short fact 1", "short fact 2"],
      "boxes": [
        { "x_min": 0.12, "y_min": 0.34, "x_max": 0.56, "y_max": 0.78 }
      ]
    }
  ]
}
Rules:
- ONLY use the three allowed categories: "person", "animal", "object". Never invent others.
- The "boxes" array MUST contain one box per visible instance (length must equal "count").
- Be specific for animals — prefer "Golden Retriever" over "dog".
- If nothing in the allowed categories is visible, return { "summary": "...", "subjects": [] }.`
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
